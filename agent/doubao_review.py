"""
doubao_review.py
~~~~~~~~~~~~~~
使用字节跳动豆包（Doubao）大模型对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，通过火山引擎方舟 OpenAI 兼容接口调用。

用法：
    python agent/doubao_review.py
    python agent/doubao_review.py --config config/doubao_review.yaml

配置：
    默认读取 config/doubao_review.yaml。

环境变量：
    DOUBAO_API_KEY  —— 豆包 API Key（必填）

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import os
import sys
from pathlib import Path
from typing import Any

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
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "doubao_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Doubao 模型参数
    "model": "doubao-seed-2.0-pro",
    "request_delay": 1,  # 多线程版本请求间隔降低到1秒
    "max_workers": 5,    # 最大并发数，根据API限流调整，默认5并发
    "skip_existing": False,
    "suggest_min_score": 4.0,
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


class DoubaoReviewer(BaseReviewer):
    def __init__(self, config):
        super().__init__(config)

        api_key = os.environ.get("DOUBAO_API_KEY", "")
        if not api_key:
            print("[ERROR] 未找到环境变量 DOUBAO_API_KEY，请先设置后重试。", file=sys.stderr)
            sys.exit(1)

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3",
        )

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
        调用 Doubao API，对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        user_text = (
            f"股票代码：{code}\n\n"
            "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
            "并严格按照要求输出 JSON。"
        )

        b64_data, mime_type = self.image_to_base64(day_chart)

        response = self.client.chat.completions.create(
            model=self.config.get("model", "doubao-seed-2.0-pro"),
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
            raise RuntimeError(f"Doubao 返回空响应，无法解析 JSON（code={code}）")

        result = self.extract_json(response_text)
        result["code"] = code
        return result


def main():
    parser = argparse.ArgumentParser(description="Doubao 图表复评")
    parser.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG_PATH),
        help="配置文件路径（默认 config/doubao_review.yaml）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    reviewer = DoubaoReviewer(config)  # 修复：原来是 QwenReviewer，现在改为 DoubaoReviewer
    reviewer.run()


if __name__ == "__main__":
    main()