"""
doubao_batch_review.py
~~~~~~~~~~~~~~~~~~~~~

使用字节跳动豆包（Doubao）大模型对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，通过火山引擎方舟 OpenAI 兼容接口调用。

简化版批量处理：
    - 分批处理股票分析请求（最大10个一批）
    - 批次间添加延迟，避免API限制
    - 实时处理，无需续传功能

重要说明：
    每张图像都会被单独分析和打分，每个股票结果保存到独立 JSON 文件。

用法：
    # 豆包 API 模式
    export DOUBAO_API_KEY="your-api-key"
    python agent/doubao_batch_review.py
    python agent/doubao_batch_review.py --config config/doubao_review.yaml

配置：
    默认读取 config/doubao_review.yaml。
    可通过命令行参数覆盖配置。

环境变量：
    DOUBAO_API_KEY  —— 豆包 API Key（使用 API 模式时必填）

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
from pathlib import Path
from typing import Any, List, Dict
from openai import OpenAI
import yaml

try:
    from .base_reviewer import BaseReviewer
except ImportError:
    from base_reviewer import BaseReviewer

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
    # Doubao 模型参数
    "model": "doubao-seed-2.0-pro",
    "request_delay": 1,  # 请求间隔（秒），默认 1 秒，避免请求过于密集
    "max_workers": 1,      # 默认单线程，避免被 API 限流
    "skip_existing": True,
    "suggest_min_score": 4.0,
    # 批量处理参数
    "batch_size": 10,      # 批量处理大小
    "batch_delay": 60,     # 批量处理间隔（秒）
}

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or (_ROOT / "config" / "doubao_review.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


class BatchDoubaoReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)

        # 批量处理参数
        self.batch_size = config.get("batch_size", 10)
        self.batch_delay = config.get("batch_delay", 60)

        # 初始化 Doubao 客户端
        api_key = os.environ.get("DOUBAO_API_KEY", "")
        if not api_key:
            print("[ERROR] 未找到环境变量 DOUBAO_API_KEY，请先设置后重试。", file=sys.stderr)
            sys.exit(1)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        )
        self.model = config.get("model", "doubao-seed-2.0-pro")
        print(f"[INFO] 已启用豆包 API 模式（实时分批处理）")
        print(f"       模型名称：{self.model}")
        print(f"       Base URL: https://ark.cn-beijing.volces.com/api/coding/v3")

    @staticmethod
    def image_to_base64(path: Path) -> tuple[str, str]:
        """将图片文件转为 base64 字符串及对应 mime_type。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8"), mime_type

    def process_batch(self, batch_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理一批股票（最大10个）

        Args:
            batch_items: 包含股票信息的列表

        Returns:
            结果字典，key 为股票代码，value 为分析结果
        """
        results = {}

        for item in batch_items:
            code = item["code"]
            day_chart = item["day_chart"]
            prompt = item["prompt"]

            try:
                user_text = (
                    f"股票代码：{code}\n\n"
                    "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
                    "并严格按照要求输出 JSON。"
                )

                b64_data, mime_type = self.image_to_base64(day_chart)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
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
                    temperature=0.2,
                )

                response_text = response.choices[0].message.content
                if response_text is None:
                    raise RuntimeError(f"Doubao 返回空响应（code={code}）")

                result = self.extract_json(response_text)
                result["code"] = code
                results[code] = result

                # 立即打印详细信息
                verdict = result.get("verdict", "?")
                score = result.get("total_score", "?")
                print(f"[✅] {code} — 完成 | verdict={verdict}, score={score}")

            except Exception as e:
                print(f"[❌] {code} — 失败: {str(e)}")
                results[code] = {"error": str(e), "code": code}

        return results

    def run(self):
        """
        重构后的 run 方法：
        1. 分批累积请求（最大10个一批）
        2. 实时处理，无需续传
        3. 统一保存结果

        保证每张图单独获得大模型打分结果
        """
        # 加载候选股票
        candidates_data = self.load_candidates(Path(self.config["candidates"]))
        pick_date: str = candidates_data["pick_date"]
        candidates: List[dict] = candidates_data["candidates"]
        print(f"[INFO] pick_date={pick_date}，候选股票数={len(candidates)}")

        out_dir = self.output_dir / pick_date
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: List[dict] = []
        failed_codes: List[str] = []

        # 步骤 1：处理已经存在的缓存文件
        cached_count = 0
        processed_codes = set()
        for candidate in candidates:
            code = candidate["code"]
            out_file = out_dir / f"{code}.json"
            if self.config.get("skip_existing", False) and out_file.exists():
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
        print(f"[INFO] 待分析股票数：{len(to_process)} (已缓存 {cached_count} 支)")

        if to_process:
            # 步骤 3：分批处理请求
            batch_size = self.config.get("batch_size", 10)
            batch_count = 0

            for i in range(0, len(to_process), batch_size):
                batch = to_process[i:i + batch_size]
                batch_count += 1
                print(f"[INFO] 开始处理第 {batch_count} 批，包含 {len(batch)} 个股票...")

                # 准备当前批次的数据
                batch_items = []
                for candidate in batch:
                    code = candidate["code"]
                    # 找到对应的图表
                    day_chart = self.find_chart_images(pick_date, code)
                    if day_chart is None:
                        failed_codes.append(code)
                        print(f"[WARN] {code} — 未找到图表文件，跳过")
                        continue

                    # 添加到批次数据
                    batch_items.append({
                        "code": code,
                        "day_chart": day_chart,
                        "prompt": self.prompt
                    })

                # 步骤 4：处理当前批次
                if batch_items:
                    print(f"[INFO] 提交第 {batch_count} 批，包含 {len(batch_items)} 个请求...")
                    batch_results = self.process_batch(batch_items)

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


def main():
    parser = argparse.ArgumentParser(description="Doubao 图表复评 - 简化批量处理模式（支持豆包 API）")
    parser.add_argument(
        "--config",
        default="config/doubao_review.yaml",
        help="配置文件路径（默认 config/doubao_review.yaml）",
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
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # 命令行参数覆盖配置
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.batch_delay is not None:
        config["batch_delay"] = args.batch_delay

    reviewer = BatchDoubaoReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()