"""
qwen_review.py
~~~~~~~~~~~~~~
使用阿里云通义千问（Qwen）大模型对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，通过 DashScope 兼容接口调用。

并发控制：
    - 支持单线程/多线程模式
    - 内置 Qwen API 限流保护（RPM/TPM/RPS）
    - 自动重试和错误处理

用法：
    # 单线程模式
    python agent/qwen_review.py
    python agent/qwen_review.py --config config/qwen_review.yaml

    # 多线程模式（推荐）
    python agent/qwen_review.py --max-workers 10
    python agent/qwen_review.py --max-workers 20 --request-delay 0.5

配置：
    默认读取 config/qwen_review.yaml。
    可通过命令行参数覆盖配置。

环境变量：
    DASHSCOPE_API_KEY  —— 阿里云 DashScope API Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
import yaml

try:
    from .base_reviewer import BaseReviewer
    from .qwen_concurrency import (
        QwenConcurrencyController,
        STANDARD_CONFIG,
        CONSERVATIVE_CONFIG,
    )
except ImportError:
    from base_reviewer import BaseReviewer
    from qwen_concurrency import (
        QwenConcurrencyController,
        STANDARD_CONFIG,
        CONSERVATIVE_CONFIG,
    )

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "qwen_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Qwen 模型参数
    "model": "qwen3.5-plus",
    "request_delay": 0.5,  # 请求间隔（秒），多线程时建议设置 0.5-1
    "max_workers": 10,     # 最大并发数，根据 API 限流调整
    "skip_existing": True,
    "suggest_min_score": 4.0,
    # 并发控制
    "use_rate_limit": True,      # 是否启用限流保护
    "rate_limit_profile": "standard",  # conservative/standard/aggressive
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    cfg_path = config_path or _DEFAULT_CONFIG_PATH
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


class QwenReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            print("[ERROR] 未找到环境变量 DASHSCOPE_API_KEY，请先设置后重试。", file=sys.stderr)
            sys.exit(1)

        # 使用阿里云百炼平台的 API 接口
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
        )

        # 初始化并发控制器
        self.controller = None
        if config.get("use_rate_limit", True):
            profile = config.get("rate_limit_profile", "standard")
            if profile == "conservative":
                self.controller = QwenConcurrencyController(CONSERVATIVE_CONFIG)
            else:
                self.controller = QwenConcurrencyController(STANDARD_CONFIG)
            print(f"[INFO] 并发限流保护已启用（配置档位：{profile}）")
            print(f"       最大并发数：{self.controller.config.max_concurrent_workers}")
            print(f"       RPM 上限：{self.controller.config.safe_rpm}")
            print(f"       TPM 上限：{self.controller.config.safe_tpm}")
        else:
            print("[WARN] 并发限流保护未启用，请自行控制请求频率")

    @staticmethod
    def image_to_base64(path: Path) -> tuple[str, str]:
        """将图片文件转为 base64 字符串及对应 mime_type。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8"), mime_type

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用 Qwen API，对单支股票进行图表分析，返回解析后的 JSON 结果。

        集成并发控制：
        - 自动获取 API 许可（受 RPM/TPM 限制）
        - 失败自动重试（最多 3 次）
        - 记录实际 token 消耗
        """
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )

        b64_data, mime_type = self.image_to_base64(day_chart)

        # 获取并发许可（会受 RPM/TPM 限流控制）
        if self.controller:
            estimated_tokens = self.controller.config.estimated_tokens_per_request
            if not self.controller.acquire(estimated_tokens=estimated_tokens, timeout=300):
                raise TimeoutError(f"获取 API 许可超时（code={code}），请稍后重试")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.monotonic()

                response = self.client.chat.completions.create(
                    model=self.config.get("model", "qwen3.5-plus"),
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

                # 记录成功（更新实际 token 消耗）
                if self.controller:
                    self.controller.record_success(actual_tokens=actual_tokens)

                return result

            except Exception as e:
                last_error = e
                if self.controller:
                    self.controller.record_failure()

                # 指数退避重试
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s
                    print(f"[⚠️ ] {code} — 请求失败 ({e})，{wait_time}秒后重试 ({attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"{code} — 重试{max_retries}次后仍失败：{e}")

        raise RuntimeError(f"{code} — 重试耗尽：{last_error}")

    def run(self):
        """
        重写 run 方法，在结束时打印并发统计
        """
        # 调用父类的 run
        super().run()

        # 打印统计信息
        if self.controller:
            self.controller.print_stats()


def main():
    parser = argparse.ArgumentParser(description="Qwen 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/qwen_review.yaml）",
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
    args = parser.parse_args()

    config = load_config(Path(args.config))

    # 命令行参数覆盖配置
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers
    if args.request_delay is not None:
        config["request_delay"] = args.request_delay
    if args.no_rate_limit:
        config["use_rate_limit"] = False
    if args.rate_limit_profile is not None:
        config["rate_limit_profile"] = args.rate_limit_profile

    reviewer = QwenReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
