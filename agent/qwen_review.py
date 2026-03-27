"""
qwen_review.py
~~~~~~~~~~~~~~
使用 Qwen 大模型（阿里云 API 或 Ollama 本地部署）对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，通过 OpenAI 兼容接口调用。

并发控制：
    - 支持单线程/多线程模式
    - 内置 Qwen API 限流保护（RPM/TPM/RPS）
    - 自动重试和错误处理

用法：
    # 阿里云 API 模式
    export DASHSCOPE_API_KEY="your-api-key"
    python agent/qwen_review.py
    python agent/qwen_review.py --config config/qwen_review.yaml

    # Ollama 本地模式
    python agent/qwen_review.py --use-ollama
    python agent/qwen_review.py --use-ollama --model qwen3-vl:8b

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
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "qwen_review.yaml"

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
    # 批量处理参数（默认启用批量模式以提高效率）
    "batch_mode": True,   # 是否启用批量处理模式，默认开启
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

        # 判断是否使用 Ollama 本地模型
        use_ollama = config.get("use_ollama", False)

        if use_ollama:
            # Ollama 本地部署模式
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
                print(f"[INFO] 已启用 Ollama 本地模型模式")
                print(f"       模型名称：{ollama_model}")
                print(f"       服务地址：{ollama_url}")
                print(f"       最大并发数：{self.controller.config.max_concurrent_workers}")
                print(f"       请求间隔：{self.controller.config.request_delay}秒")
            else:
                self.controller = None
                print(f"[INFO] 已启用 Ollama 本地模型模式（无并发限流）")
                print(f"       模型名称：{ollama_model}")
                print(f"       服务地址：{ollama_url}")
                print(f"       提示：Ollama 本地运行无需 API 限流保护")
        else:
            # 阿里云 API 模式
            api_key = os.environ.get("DASHSCOPE_API_KEY", "")
            if not api_key:
                print("[ERROR] 未找到环境变量 DASHSCOPE_API_KEY，请先设置后重试。", file=sys.stderr)
                print("[提示] 如需使用 Ollama 本地模型，请添加 --use-ollama 参数")
                sys.exit(1)

            self.client = OpenAI(
                api_key=api_key,
                # base_url="https://coding.dashscope.aliyuncs.com/v1",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = config.get("model", "qwen3-vl-plus")
            print(f"[INFO] 已启用阿里云 API 模式")
            print(f"       模型名称：{self.model}")

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
        调用 Qwen 模型（API 或 Ollama 本地），对单支股票进行图表分析，返回解析后的 JSON 结果。

        Ollama 本地模式：
        - 无 API 限流，但需控制请求频率避免本地资源过载
        - 使用 OllamaConcurrencyController 追踪进度

        阿里云 API 模式：
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

        # 获取并发许可（Ollama 用信号量，API 用 RPM/TPM）
        if self.controller:
            # 判断控制器类型
            if isinstance(self.controller, OllamaConcurrencyController):
                # Ollama 模式：信号量控制
                if not self.controller.acquire(timeout=300):
                    raise TimeoutError(f"获取并发许可超时（code={code}），请稍后重试")
            else:
                # API 模式：RPM/TPM 控制
                estimated_tokens = self.controller.config.estimated_tokens_per_request
                if not self.controller.acquire(estimated_tokens=estimated_tokens, timeout=300):
                    raise TimeoutError(f"获取 API 许可超时（code={code}），请稍后重试")

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                start_time = time.monotonic()

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

                elapsed = time.monotonic() - start_time

                # 获取实际 token 消耗（Ollama 可能不返回）
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
                    if isinstance(self.controller, OllamaConcurrencyController):
                        self.controller.record_success()
                        self.controller.release()  # Ollama 需要手动释放
                    else:
                        self.controller.record_success(actual_tokens=actual_tokens)

                return result

            except Exception as e:
                last_error = e
                if self.controller:
                    if isinstance(self.controller, OllamaConcurrencyController):
                        self.controller.record_failure()
                        self.controller.release()  # Ollama 需要手动释放
                    else:
                        self.controller.record_failure()

                # 指数退避重试
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # 1s, 2s, 4s
                    print(f"\n[⚠️ ] {code} — 请求失败 ({e})，{wait_time}秒后重试 ({attempt+1}/{max_retries})")
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
    parser = argparse.ArgumentParser(description="Qwen 图表复评（支持阿里云 API 和 Ollama 本地模型）")
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
    # 批量处理参数
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="启用批量处理模式（默认已启用）",
    )
    parser.add_argument(
        "--no-batch-mode",
        action="store_true",
        help="禁用批量处理模式，使用普通模式",
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
    if args.max_workers is not None:
        config["max_workers"] = args.max_workers
    if args.request_delay is not None:
        config["request_delay"] = args.request_delay
    if args.no_rate_limit:
        config["use_rate_limit"] = False
    if args.rate_limit_profile is not None:
        config["rate_limit_profile"] = args.rate_limit_profile
    # 批量处理参数覆盖
    if args.no_batch_mode:
        config["batch_mode"] = False
    elif args.batch_mode:
        config["batch_mode"] = True
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.batch_delay is not None:
        config["batch_delay"] = args.batch_delay
    # Ollama 参数覆盖
    if args.use_ollama:
        config["use_ollama"] = True
    if args.model is not None:
        config["ollama_model"] = args.model
    if args.ollama_url is not None:
        config["ollama_base_url"] = args.ollama_url

    # 根据配置决定使用哪种处理器
    if config.get("batch_mode", True):  # 默认为True
        # 动态导入批量处理模块
        try:
            from .qwen_batch_review import BatchQwenReviewer
        except ImportError:
            from qwen_batch_review import BatchQwenReviewer

        reviewer = BatchQwenReviewer(config)
    else:
        reviewer = QwenReviewer(config)

    reviewer.run()


if __name__ == "__main__":
    main()
