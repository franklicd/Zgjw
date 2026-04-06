"""
qwen_batch_review.py
~~~~~~~~~~~~~~~~~~~~
使用 Qwen 大模型的批量处理模式对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，通过 OpenAI 兼容接口调用批量 API。

批量控制：
    - 使用阿里百炼 Batch API 最佳实践
    - 上传 JSONL 文件，批量处理所有请求
    - 异步轮询，完成后下载结果
    - 内置 Qwen API 限流保护（RPM/TPM/RPS）
    - 自动重试和错误处理

重要说明：
    每张图像都会被单独分析和打分，使用 custom_id = code 确保唯一标识，
    结果返回后按 custom_id 匹配，每个股票结果保存到独立 JSON 文件。

用法：
    # 阿里云 API 模式
    export DASHSCOPE_API_KEY="your-api-key"
    python agent/qwen_batch_review.py
    python agent/qwen_batch_review.py --config config/qwen_review.yaml

    # Ollama 本地模式（注意：Ollama 不支持真正的批量 API）
    python agent/qwen_batch_review.py --use-ollama

配置：
    默认读取 config/qwen_review.yaml。
    可通过命令行参数覆盖配置。

环境变量：
    DASHSCOPE_API_KEY  —— 阿里云 DashScope API Key（使用 API 模式时必填）
    OLLAMA_BASE_URL    —— Ollama 服务地址（可选，默认 http://localhost:11434/v1）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import os
import sys
import time
import json
import tempfile
from pathlib import Path
from typing import Any, List, Dict
from openai import OpenAI
import yaml

try:
    from .base_reviewer import BaseReviewer
    from .qwen_concurrency import (
        QwenConcurrencyController,
        STANDARD_CONFIG,
        CONSERVATIVE_CONFIG,
    )
    from .ollama_concurrency import (
        OllamaConcurrencyController,
        STANDARD_CONFIG as OLLAMA_STANDARD_CONFIG,
    )
except ImportError:
    from base_reviewer import BaseReviewer
    from qwen_concurrency import (
        QwenConcurrencyController,
        STANDARD_CONFIG,
        CONSERVATIVE_CONFIG,
    )
    from ollama_concurrency import (
        OllamaConcurrencyController,
        STANDARD_CONFIG as OLLAMA_STANDARD_CONFIG,
    )

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Qwen 模型参数
    "model": "qwen3-vl-plus",
    "request_delay": 1,  # 请求间隔（秒），默认 1 秒，避免请求过于密集
    "max_workers": 1,      # 默认单线程，避免被 API 限流
    "skip_existing": True,
    "suggest_min_score": 4.0,
    # 批量处理参数
    "batch_size": 10,      # 批量处理大小
    "batch_delay": 60,     # 批量处理间隔（秒）
    # 并发控制
    "use_rate_limit": True,      # 是否启用限流保护
    "rate_limit_profile": "standard",  # conservative/standard/aggressive
    # Ollama 本地部署配置
    "use_ollama": False,           # 是否使用 Ollama 本地模型
    "ollama_base_url": "http://localhost:11434/v1",  # Ollama 服务地址
    "ollama_model": "qwen3-vl:8b",  # Ollama 本地模型名称
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    from .qwen_review import load_config as qwen_load_config  # 使用 qwen_review 的配置加载
    return qwen_load_config(config_path)


class BatchQwenReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)

        # 批量处理参数
        self.batch_size = config.get("batch_size", 10)
        self.batch_delay = config.get("batch_delay", 60)  # 批量处理间隔

        # 判断是否使用 Ollama 本地模型
        use_ollama = config.get("use_ollama", False)

        if use_ollama:
            # Ollama 本地部署模式 - 注意：Ollama 不支持真正的批量 API，使用降级方案
            ollama_url = os.environ.get("OLLAMA_BASE_URL", config.get("ollama_base_url", "http://localhost:11434/v1"))
            ollama_model = config.get("ollama_model", "qwen3-vl:8b")
            self.client = OpenAI(
                api_key="ollama",  # Ollama 不需要真实 key
                base_url=ollama_url,
            )
            self.model = ollama_model

            # Ollama 并发控制器
            use_rate_limit = config.get("use_rate_limit", False)  # Ollama 默认不限流
            if use_rate_limit:
                self.controller = OllamaConcurrencyController(OLLAMA_STANDARD_CONFIG)
                print(f"[INFO] 已启用 Ollama 本地模型模式（批量处理）")
                print(f"       模型名称：{ollama_model}")
                print(f"       服务地址：{ollama_url}")
                print(f"       最大并发数：{self.controller.config.max_concurrent_workers}")
                print(f"       请求间隔：{self.controller.config.request_delay}秒")
            else:
                self.controller = None
                print(f"[INFO] 已启用 Ollama 本地模型模式（无并发限流，批量处理）")
                print(f"       模型名称：{ollama_model}")
                print(f"       服务地址：{ollama_url}")
                print(f"       提示：Ollama 本地运行无需 API 限流保护")
        else:
            # 阿里云 API 模式 - 支持批量处理
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                print("[ERROR] 未找到环境变量 DASHSCOPE_API_KEY，请先设置后重试。", file=sys.stderr)
                print("[提示] 如需使用 Ollama 本地模型，请添加 --use-ollama 参数")
                sys.exit(1)

            # 使用阿里云兼容 OpenAI 的批量接口
            # 注意：Batch API 使用独立的 endpoint
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = config.get("model", "qwen3-vl-plus")
            print(f"[INFO] 已启用阿里云 API 模式（批量处理）")
            print(f"       模型名称：{self.model}")
            print(f"       Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1")

            # 初始化并发控制器
            if config.get("use_rate_limit", True):
                profile = config.get("rate_limit_profile", "standard")
                if profile == "conservative":
                    self.controller = QwenConcurrencyController(CONSERVATIVE_CONFIG)
                else:
                    self.controller = QwenConcurrencyController(STANDARD_CONFIG)
                print(f"       并发限流保护已启用（配置档位：{profile}）")
                print(f"       最大并发数：{self.controller.config.max_concurrent_workers}")
                print(f"       RPM 上限：{self.controller.config.safe_rpm}")
                print(f"       TPM 上限：{self.controller.config.safe_tpm}")
            else:
                print("[WARN] 并发限流保护未启用，请自行控制请求频率")
                self.controller = None

        # 初始化批量请求缓存
        self.batch_requests = []
        self.results_cache = {}

        # 初始化批处理状态跟踪
        self.batch_status_dir = _ROOT / "data" / "batch_status"
        self.batch_status_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def image_to_base64(path: Path) -> tuple[str, str]:
        """将图片文件转为 base64 字符串及对应 mime_type。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8"), mime_type

    def build_batch_request(self, code: str, day_chart: Path, prompt: str) -> Dict[str, Any]:
        """
        构建单个股票分析请求，符合阿里 Batch API 格式
        
        Args:
            code: 股票代码，用作 custom_id 确保唯一标识
            day_chart: K 线图表路径
            prompt: 系统提示词
            
        Returns:
            符合 Batch API 输入文件格式的请求对象
        """
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )

        b64_data, mime_type = self.image_to_base64(day_chart)

        # 构造消息对象，符合阿里 Batch API 格式
        # custom_id 使用股票代码，确保每个请求有唯一标识
        request = {
            "custom_id": code,
            "method": "POST",
            "url": "/v1/chat/completions",  # endpoint 必须与创建 batch 时一致
            "body": {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "【日线图】"},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{b64_data}"
                                },
                            },
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
                "temperature": 0.2,
            }
        }

        return request

    def submit_batch(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        提交批量请求到阿里百炼 Batch API

        遵循最佳实践：
        1. 上传 JSONL 文件
        2. 创建批量作业
        3. 异步轮询状态
        4. 下载结果文件

        Args:
            requests: 批量请求列表，每个请求包含 custom_id, method, url, body

        Returns:
            结果字典，key 为 custom_id（股票代码），value 为分析结果
        """
        if not requests:
            print("[INFO] 批量请求为空，跳过提交")
            return {}

        print(f"[INFO] 准备提交包含 {len(requests)} 个请求的批量作业...")

        import io

        # 步骤 0：检查挂起的批量作业状态
        pending_results = self._check_existing_batches_status()
        if pending_results:
            print(f"[INFO] 从挂起的批量作业中恢复了 {len(pending_results)} 个结果")

        # 步骤 1：创建 JSONL 文件内容
        # 每行一个 JSON 对象，符合 Batch API 输入文件格式
        jsonl_content = ""
        for req in requests:
            jsonl_content += json.dumps(req, ensure_ascii=False) + "\n"

        # 将字符串内容转换为字节
        jsonl_bytes = jsonl_content.encode('utf-8')

        # 步骤 2：上传输入文件到阿里百炼
        try:
            # 创建 BytesIO 对象
            file_io = io.BytesIO(jsonl_bytes)
            file_io.name = "batch_input.jsonl"  # 设置文件名

            uploaded_file = self.client.files.create(
                file=file_io,
                purpose="batch"
            )

            print(f"[INFO] ✓ 文件上传成功，文件 ID: {uploaded_file.id}")
        except Exception as e:
            print(f"[❌] 上传批量输入文件失败：{e}")
            print("[INFO] 降级到并行处理模式...")
            return self._process_sequentially(requests)

        batch_id = None
        try:
            # 步骤 3：创建批量作业
            # endpoint 必须与输入文件中的 url 字段保持一致
            batch_job = self.client.batches.create(
                input_file_id=uploaded_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",  # 24 小时内完成
                metadata={
                    "description": f"Qwen stock analysis batch for {len(requests)} stocks"
                }
            )

            batch_id = batch_job.id
            print(f"[INFO] ✓ 批量作业已创建，作业 ID: {batch_id}")

            # 保存批量作业信息以便恢复
            job_info = {
                "batch_id": batch_id,
                "request_count": len(requests),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "created"
            }
            self._save_batch_job_info(batch_id, job_info)

            # 步骤 4：轮询批量作业状态直到完成
            # 可能的状态：validating → finalizing → processing/in_progress → completed/failed/expired/cancelled
            poll_count = 0
            start_time = time.time()  # 记录开始等待时间
            max_wait_time = 24 * 3600  # 最大等待时间：24小时（批处理窗口）
            check_interval = 60  # 检查间隔：60秒（更合适的轮询间隔）

            while True:
                poll_count += 1
                elapsed_time = time.time() - start_time  # 计算已等待时间

                # 如果超过最大等待时间，退出循环
                if elapsed_time > max_wait_time:
                    print(f"[❌] 批量作业等待超时（24小时），状态：{batch_job.status}")
                    batch_job.status = "timed_out"
                    break

                # 显示进度提示：轮询次数、当前状态、已等待时间
                print(f"[INFO] 轮询进度 [{poll_count}] | 状态：{batch_job.status} | 已等待：{elapsed_time:.0f}秒")

                # 检查是否完成
                if batch_job.status == "completed":
                    elapsed_total = time.time() - start_time
                    print(f"[✅] 批量作业已完成，处理了 {batch_job.request_counts.total} 个请求 | 总耗时：{elapsed_total:.0f}秒")
                    break
                elif batch_job.status in ["failed"]:
                    elapsed_total = time.time() - start_time
                    print(f"[❌] 批量作业失败：{batch_job.error} | 已等待：{elapsed_total:.0f}秒")
                    break
                elif batch_job.status in ["cancelled"]:
                    elapsed_total = time.time() - start_time
                    print(f"[❌] 批量作业已被取消 | 已等待：{elapsed_total:.0f}秒")
                    break
                elif batch_job.status in ["expired"]:
                    elapsed_total = time.time() - start_time
                    print(f"[❌] 批量作业已过期 | 已等待：{elapsed_total:.0f}秒")
                    break
                elif batch_job.status in ["in_progress", "validating", "finalizing", "processing", "cancelling"]:
                    # 这些状态表示仍在进行中，需要继续等待
                    time.sleep(check_interval)  # 等待后继续检查
                    # 获取最新的作业状态
                    batch_job = self.client.batches.retrieve(batch_id)
                    continue
                else:
                    # 未知状态，也认为需要降级处理
                    elapsed_total = time.time() - start_time
                    print(f"[❌] 批量作业遇到未知状态：{batch_job.status} | 已等待：{elapsed_total:.0f}秒")
                    break

            # 步骤 5：处理批量作业结果
            results = {}
            if batch_job.status == "completed" and batch_job.output_file_id:
                try:
                    # 下载输出文件
                    output_file_content = self.client.files.content(batch_job.output_file_id)

                    # 解析输出文件内容
                    # 输出文件格式：每行一个 JSON 对象，包含 id, custom_id, response 等字段
                    success_count = 0
                    error_count = 0

                    # 检查内容类型并适当地处理
                    content = output_file_content.content
                    if isinstance(content, bytes):
                        # 如果是字节内容，则解码
                        content_str = content.decode('utf-8')
                    elif hasattr(output_file_content, 'text') and output_file_content.text:
                        # 如果有text属性，则使用text
                        content_str = output_file_content.text
                    else:
                        # 否则是直接的字符串
                        content_str = content if isinstance(content, str) else str(content)

                    for line in content_str.strip().split('\n'):
                        if line.strip():
                            output_item = json.loads(line)
                            custom_id = output_item.get('custom_id')

                            if 'response' in output_item:
                                response_body = output_item['response'].get('body', {})
                                response_text = response_body.get('choices', [{}])[0].get('message', {}).get('content')

                                if response_text:
                                    result = self.extract_json(response_text)
                                    result["code"] = custom_id
                                    results[custom_id] = result
                                    success_count += 1
                                else:
                                    results[custom_id] = {"error": "Empty response", "code": custom_id}
                                    error_count += 1
                            else:
                                results[custom_id] = {"error": "No response in batch output", "code": custom_id}
                                error_count += 1

                    print(f"[INFO] 结果解析完成：成功 {success_count} 个，失败 {error_count} 个")

                except Exception as e:
                    print(f"[❌] 处理批量输出文件时出错：{e}")
                    # 降级处理
                    return self._process_sequentially(requests)
            elif batch_job.status == "failed":
                print(f"[❌] 批量作业失败，降级到并行处理模式")
                return self._process_sequentially(requests)
            else:
                print(f"[❌] 批量作业未完成或无输出文件（状态：{batch_job.status}），降级到并行处理模式")
                return self._process_sequentially(requests)

        except Exception as e:
            print(f"[❌] 批量 API 调用失败：{e}")
            print("[INFO] 降级到并行处理模式...")
            return self._process_sequentially(requests)
        finally:
            # 清理上传的输入文件（可选，避免存储费用）
            try:
                if 'uploaded_file' in locals():
                    self.client.files.delete(uploaded_file.id)
                    print(f"[INFO] 已清理临时输入文件：{uploaded_file.id}")
            except:
                pass

            # 如果批量作业成功完成，删除状态文件
            if batch_id and batch_job and batch_job.status == "completed":
                self._delete_batch_job_info(batch_id)

        # 合并挂起作业的结果和本次作业的结果
        all_results = pending_results.copy()
        all_results.update(results)

        return all_results

    def _process_sequentially(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        降级方案：当批量 API 不可用时，使用顺序处理
        
        Args:
            requests: 批量请求列表
            
        Returns:
            结果字典
        """
        print(f"[INFO] 使用顺序处理降级方案...")

        results = {}

        for req in requests:
            code = req["custom_id"]
            try:
                # 检查并发控制器
                if self.controller:
                    estimated_tokens = self.controller.config.estimated_tokens_per_request
                    if not self.controller.acquire(estimated_tokens=estimated_tokens, timeout=300):
                        raise TimeoutError(f"获取 API 许可超时（code={code}），请稍后重试")

                start_time = time.monotonic()

                response = self.client.chat.completions.create(
                    model=req["body"]["model"],
                    messages=req["body"]["messages"],
                    temperature=req["body"]["temperature"],
                )

                elapsed = time.monotonic() - start_time

                # 获取实际 token 消耗
                usage = getattr(response, "usage", None)
                actual_tokens = 0
                if usage:
                    actual_tokens = getattr(usage, "total_tokens", 0)

                response_text = response.choices[0].message.content
                if response_text is None:
                    raise RuntimeError(f"Qwen 返回空响应（code={code}）")

                result = self.extract_json(response_text)
                result["code"] = code

                # 记录成功
                if self.controller:
                    self.controller.record_success(actual_tokens=actual_tokens)

                results[code] = result
                print(f"[✅] 顺序分析完成：{code}")

            except Exception as e:
                if self.controller:
                    self.controller.record_failure()

                print(f"[❌] 单独分析失败 {code}: {e}")
                results[code] = {"error": str(e), "code": code}

        return results

    def _save_batch_job_info(self, batch_id: str, job_info: dict):
        """
        保存批量作业信息到文件，以便断网重连后可以继续监控
        """
        try:
            batch_status_file = self.batch_status_dir / f"{batch_id}.json"
            with open(batch_status_file, 'w', encoding='utf-8') as f:
                json.dump(job_info, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] 保存批量作业信息失败: {e}")

    def _load_batch_job_info(self, batch_id: str) -> dict:
        """
        从文件加载批量作业信息
        """
        try:
            batch_status_file = self.batch_status_dir / f"{batch_id}.json"
            if batch_status_file.exists():
                with open(batch_status_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"[WARN] 加载批量作业信息失败: {e}")
        return {}

    def _delete_batch_job_info(self, batch_id: str):
        """
        删除批量作业信息文件
        """
        try:
            batch_status_file = self.batch_status_dir / f"{batch_id}.json"
            if batch_status_file.exists():
                batch_status_file.unlink()
        except Exception as e:
            print(f"[WARN] 删除批量作业信息失败: {e}")

    def _check_existing_batches_status(self) -> Dict[str, Any]:
        """
        检查已有批量作业的状态
        """
        results = {}
        batch_files = self.batch_status_dir.glob("batch_*.json")
        active_batch_ids = []

        for batch_file in batch_files:
            try:
                with open(batch_file, 'r', encoding='utf-8') as f:
                    job_info = json.load(f)

                batch_id = job_info.get("batch_id")
                if batch_id:
                    active_batch_ids.append(batch_id)
                    print(f"[INFO] 发现已存在的批量作业: {batch_id} (创建时间: {job_info.get('created_at', '?')})")
            except Exception as e:
                print(f"[WARN] 读取批量作业状态文件失败 {batch_file}: {e}")

        if active_batch_ids:
            print(f"[INFO] 检查 {len(active_batch_ids)} 个已存在的批量作业状态...")
            for batch_id in active_batch_ids:
                try:
                    batch_job = self.client.batches.retrieve(batch_id)
                    print(f"[INFO] 批量作业 {batch_id} 当前状态: {batch_job.status}")

                    if batch_job.status == "completed":
                        print(f"[INFO] 批量作业 {batch_id} 已完成，正在下载结果...")
                        results.update(self._download_batch_results(batch_job))
                        self._delete_batch_job_info(batch_id)
                    elif batch_job.status in ["failed", "cancelled", "expired"]:
                        print(f"[WARN] 批量作业 {batch_id} 状态异常 ({batch_job.status})，删除记录...")
                        self._delete_batch_job_info(batch_id)
                    else:
                        print(f"[INFO] 批量作业 {batch_id} 仍在处理中 ({batch_job.status})，继续监控...")

                except Exception as e:
                    print(f"[WARN] 查询批量作业 {batch_id} 状态失败: {e}")
                    # 如果无法连接，仍然保留文件供后续重试

        return results

    def _download_batch_results(self, batch_job) -> Dict[str, Any]:
        """
        下载指定批量作业的结果
        """
        results = {}
        if batch_job.output_file_id:
            try:
                # 下载输出文件
                output_file_content = self.client.files.content(batch_job.output_file_id)

                # 解析输出文件内容
                success_count = 0
                error_count = 0

                # 检查内容类型并适当地处理
                content = output_file_content.content
                if isinstance(content, bytes):
                    # 如果是字节内容，则解码
                    content_str = content.decode('utf-8')
                elif hasattr(output_file_content, 'text') and output_file_content.text:
                    # 如果有text属性，则使用text
                    content_str = output_file_content.text
                else:
                    # 否则是直接的字符串
                    content_str = content if isinstance(content, str) else str(content)

                for line in content_str.strip().split('\n'):
                    if line.strip():
                        output_item = json.loads(line)
                        custom_id = output_item.get('custom_id')

                        if 'response' in output_item:
                            response_body = output_item['response'].get('body', {})
                            response_text = response_body.get('choices', [{}])[0].get('message', {}).get('content')

                            if response_text:
                                result = self.extract_json(response_text)
                                result["code"] = custom_id
                                results[custom_id] = result
                                success_count += 1
                            else:
                                results[custom_id] = {"error": "Empty response", "code": custom_id}
                                error_count += 1
                        else:
                            results[custom_id] = {"error": "No response in batch output", "code": custom_id}
                            error_count += 1

                print(f"[INFO] 批量作业 {batch_job.id} 结果解析完成：成功 {success_count} 个，失败 {error_count} 个")

            except Exception as e:
                print(f"[❌] 处理批量输出文件时出错：{e}")
        else:
            print(f"[WARN] 批量作业 {batch_job.id} 没有输出文件")
        return results

    def run(self):
        """
        重构后的 run 方法：
        1. 检查挂起的批量作业
        2. 分批累积请求（最大10个一批）
        3. 支持断网重连的批量作业
        4. 统一保存结果

        保证每张图单独获得大模型打分结果
        """
        # 步骤 0：检查挂起的批量作业状态
        print("[INFO] 检查挂起的批量作业状态...")
        all_results_from_pending = self._check_existing_batches_status()
        if all_results_from_pending:
            print(f"[INFO] 从挂起的批量作业中恢复了 {len(all_results_from_pending)} 个结果")
        else:
            print("[INFO] 没有发现挂起的批量作业")

        # 加载候选股票
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        candidates: List[dict] = candidates_data["candidates"]
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(candidates)}")

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        failed_codes: List[str] = []

        # 添加已从挂起作业恢复的结果
        for code, result in all_results_from_pending.items():
            all_results.append(result)

        # 步骤 1：处理已经存在的缓存文件和已从挂起作业恢复的结果
        cached_count = 0
        processed_codes = set()

        # 添加已从挂起作业恢复的代码
        for code in all_results_from_pending.keys():
            processed_codes.add(code)

        for candidate in candidates:
            code = candidate["code"]
            out_file = out_dir / f"{code}.json"
            if code in processed_codes:
                # 已从挂起作业恢复的结果
                cached_count += 1
                print(f"[✅] {code} — 从挂起作业恢复，跳过")
            elif self.config.get("skip_existing", False) and out_file.exists():
                try:
                    with open(out_file, encoding="utf-8") as f:
                        result = json.load(f)
                    all_results.append(result)
                    cached_count += 1
                    processed_codes.add(code)
                    print(f"[✅] {code} — 已缓存，跳过")
                except:
                    pass

        # 步骤 2：过滤出需要处理的股票
        to_process = [c for c in candidates if c["code"] not in processed_codes]
        print(f"[INFO] 待分析股票数：{len(to_process)} (已处理/缓存 {cached_count} 支)")

        if to_process:
            # 步骤 3：分批处理请求
            batch_size = self.config.get("batch_size", 10)
            batch_count = 0

            for i in range(0, len(to_process), batch_size):
                batch = to_process[i:i + batch_size]
                batch_count += 1
                print(f"[INFO] 开始处理第 {batch_count} 批，包含 {len(batch)} 个股票...")

                # 累积当前批次的请求
                batch_requests = []

                for candidate in batch:
                    code = candidate["code"]
                    # 找到对应的图表
                    day_chart = self.find_chart_images(pick_date, code)
                    if day_chart is None:
                        failed_codes.append(code)
                        print(f"[WARN] {code} — 未找到图表文件，跳过")
                        continue

                    # 构建批量请求，custom_id = code 确保唯一标识
                    request = self.build_batch_request(
                        code=code,
                        day_chart=day_chart,
                        prompt=self.prompt,
                    )
                    batch_requests.append(request)

                # 步骤 4：提交当前批次的批量请求
                if batch_requests:
                    print(f"[INFO] 提交第 {batch_count} 批批量作业，包含 {len(batch_requests)} 个请求...")
                    batch_results = self.submit_batch(batch_requests)

                    # 步骤 5：保存当前批次的结果
                    # 每个股票结果保存到独立 JSON 文件
                    print(f"[INFO] 正在保存第 {batch_count} 批的 {len(batch_results)} 个结果...")

                    successful_count = 0
                    failed_count = 0

                    for code, result in batch_results.items():
                        # 避免重复处理
                        if code in processed_codes:
                            continue

                        out_file = out_dir / f"{code}.json"
                        with open(out_file, "w", encoding="utf-8") as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)

                        all_results.append(result)
                        verdict = result.get("verdict", "?")
                        score = result.get("total_score", "?")

                        if "error" in result:
                            print(f"[❌] {code} — 失败 | error={result['error']}")
                            failed_count += 1
                        else:
                            print(f"[✅] {code} — 完成 | verdict={verdict}, score={score}")
                            successful_count += 1
                        processed_codes.add(code)

                    print(f"[INFO] 第 {batch_count} 批处理完成：成功 {successful_count} 支，失败 {failed_count} 支")

                    # 如果不是最后一组，且设置了批量延迟，则暂停一下
                    if i + batch_size < len(to_process):
                        delay = self.config.get("batch_delay", 60)
                        if delay > 0:
                            print(f"[INFO] 等待 {delay} 秒后处理下一组...")
                            time.sleep(delay)

        print(f"\n[INFO] 评分完成：成功 {len(all_results)} 支，失败/跳过 {len(failed_codes)} 支")
        if failed_codes:
            print(f"[WARN] 未处理股票：{failed_codes}")

        if not all_results:
            print("[ERROR] 没有可用的评分结果，跳过汇总。")
            return

        # 步骤 6：生成汇总推荐建议
        print("\n[INFO] 正在生成汇总推荐建议 ...")
        min_score = self.config.get("suggest_min_score", 4.0)
        suggestion = self.generate_suggestion(
            pick_date=pick_date,
            all_results=all_results,
            min_score=min_score,
        )
        suggestion_file = out_dir / "suggestion.json"
        with open(suggestion_file, "w", encoding="utf-8") as f:
            json.dump(suggestion, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 汇总推荐已写入：{suggestion_file}")
        print(f"       推荐股票数（score≥{min_score}）: {len(suggestion['recommendations'])}")

        print("\n✅ 全部完成。")
        print(f"   输出目录：{out_dir}")

        # 打印统计信息
        if self.controller:
            self.controller.print_stats()


def main():
    parser = argparse.ArgumentParser(description="Qwen 图表复评 - 批量处理模式（支持阿里云 API）")
    parser.add_argument(
        "--config",
        default="config/qwen_review.yaml",
        help="配置文件路径（默认 config/qwen_review.yaml）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批量处理大小（覆盖配置文件设置）",
    )
    parser.add_argument(
        "--batch-delay",
        type=float,
        default=None,
        help="批量处理间隔秒数（覆盖配置文件设置）",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="最大并发数（覆盖配置文件设置）",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=None,
        help="请求间隔秒数（覆盖配置文件设置）",
    )
    parser.add_argument(
        "--no-rate-limit",
        action="store_true",
        help="禁用 API 限流保护（不推荐）",
    )
    parser.add_argument(
        "--rate-limit-profile",
        type=str,
        choices=["conservative", "standard", "aggressive"],
        default=None,
        help="限流配置档位",
    )
    # Ollama 本地模型参数
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="使用 Ollama 本地部署的模型（而非阿里云 API）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Ollama 本地模型名称（默认 qwen3-vl:8b，仅在 --use-ollama 时生效）",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=None,
        help="Ollama 服务地址（默认 http://localhost:11434/v1）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # 命令行参数覆盖配置
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.batch_delay is not None:
        config["batch_delay"] = args.batch_delay
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers
    if args.request_delay is not None:
        config["request_delay"] = args.request_delay
    if args.no_rate_limit:
        config["use_rate_limit"] = False
    if args.rate_limit_profile is not None:
        config["rate_limit_profile"] = args.rate_limit_profile
    # Ollama 参数覆盖
    if args.use_ollama:
        config["use_ollama"] = True
    if args.model is not None:
        config["ollama_model"] = args.model
    if args.ollama_url is not None:
        config["ollama_base_url"] = args.ollama_url

    reviewer = BatchQwenReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
