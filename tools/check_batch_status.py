#!/usr/bin/env python3
"""
check_batch_status.py
~~~~~~~~~~~~~~~~~~~~~

检查和管理挂起的批量作业状态。

这个脚本允许您：
- 查看所有挂起的批量作业
- 检查它们的状态
- 下载已完成作业的结果
- 清理过期或失败的作业记录
"""

import os
import json
import time
from pathlib import Path
from openai import OpenAI
import argparse

_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="检查和管理挂起的批量作业状态")
    parser.add_argument("--cleanup", action="store_true", help="清理已完成/失败的作业记录")
    parser.add_argument("--download-results", action="store_true", help="下载已完成作业的结果")

    args = parser.parse_args()

    # 初始化客户端
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print("[ERROR] 未找到环境变量 DASHSCOPE_API_KEY，请先设置后重试。")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 批量状态目录
    batch_status_dir = _ROOT / "data" / "batch_status"
    if not batch_status_dir.exists():
        print("[INFO] 未找到批量状态目录，没有挂起的作业")
        return

    # 获取所有状态文件
    batch_files = list(batch_status_dir.glob("batch_*.json"))
    if not batch_files:
        print("[INFO] 没有发现挂起的批量作业")
        return

    print(f"[INFO] 发现 {len(batch_files)} 个批量作业状态文件")

    for batch_file in batch_files:
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                job_info = json.load(f)

            batch_id = job_info.get("batch_id")
            if not batch_id:
                print(f"[WARN] 状态文件 {batch_file.name} 中没有有效的 batch_id")
                continue

            print(f"\n[INFO] 检查批量作业: {batch_id}")
            print(f"       创建时间: {job_info.get('created_at', '?')}")
            print(f"       请求数量: {job_info.get('request_count', '?')}")

            try:
                # 查询作业状态
                batch_job = client.batches.retrieve(batch_id)
                print(f"       当前状态: {batch_job.status}")

                if batch_job.status == "completed":
                    print(f"       成功处理: {getattr(batch_job.request_counts, 'completed', '?')}/{batch_job.request_counts.total} 请求")
                    if args.download_results:
                        download_batch_results(client, batch_job)
                        if args.cleanup:
                            print(f"       清理状态文件: {batch_file.name}")
                            batch_file.unlink()
                elif batch_job.status in ["failed", "cancelled", "expired"]:
                    print(f"       错误信息: {getattr(batch_job, 'error', 'None')}")
                    if args.cleanup:
                        print(f"       清理状态文件: {batch_file.name}")
                        batch_file.unlink()
                else:
                    print(f"       进度: {getattr(batch_job.request_counts, 'completed', 0)}/{batch_job.request_counts.total}")

            except Exception as e:
                print(f"       [ERROR] 查询作业状态失败: {e}")
                print(f"       提示: 可能是网络问题或作业已过期")

        except Exception as e:
            print(f"[ERROR] 读取状态文件 {batch_file.name} 失败: {e}")

def download_batch_results(client, batch_job):
    """下载指定批量作业的结果"""
    try:
        if not batch_job.output_file_id:
            print(f"       [WARN] 批量作业 {batch_job.id} 没有输出文件")
            return

        # 下载输出文件
        output_file_content = client.files.content(batch_job.output_file_id)

        # 解析输出文件内容
        content = output_file_content.content
        if isinstance(content, bytes):
            content_str = content.decode('utf-8')
        elif hasattr(output_file_content, 'text') and output_file_content.text:
            content_str = output_file_content.text
        else:
            content_str = content if isinstance(content, str) else str(content)

        # 为每个结果创建单独的文件
        results_dir = _ROOT / "data" / "review" / "batch_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for line in content_str.strip().split('\n'):
            if line.strip():
                try:
                    output_item = json.loads(line)
                    custom_id = output_item.get('custom_id')

                    if 'response' in output_item:
                        response_body = output_item['response'].get('body', {})
                        response_text = response_body.get('choices', [{}])[0].get('message', {}).get('content')

                        if response_text:
                            # 这里可以解析具体的JSON结果
                            result_file = results_dir / f"{custom_id}_batch_result.json"
                            with open(result_file, 'w', encoding='utf-8') as f:
                                f.write(response_text)
                            success_count += 1
                except json.JSONDecodeError:
                    continue

        print(f"       [INFO] 下载了 {success_count} 个结果文件到: {results_dir}")

    except Exception as e:
        print(f"       [ERROR] 下载结果失败: {e}")


if __name__ == "__main__":
    main()