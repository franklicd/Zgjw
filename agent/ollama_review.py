"""
ollama_review.py
~~~~~~~~~~~~~~~~
使用本地 Ollama 多模态大模型对候选股票进行图表分析评分。
继承自 BaseReviewer 基础架构，接口与 gemini_review.py 完全一致。

前置条件：
    1. 安装 Ollama：https://ollama.com
    2. 拉取多模态模型（任选其一）：
         ollama pull qwen2.5vl:7b      # 推荐，中文强，~6GB
         ollama pull llava:13b          # 备选，~8GB
    3. 确保 Ollama 服务已启动（ollama serve）

用法：
    python agent/ollama_review.py
    python agent/ollama_review.py --config config/ollama_review.yaml

配置：
    默认读取 config/ollama_review.yaml。
    若该文件不存在，则使用 config/gemini_review.yaml 中的通用字段。

输出：
    ./data/review/{pick_date}/{code}.json   每支股票的评分 JSON
    ./data/review/{pick_date}/suggestion.json  汇总推荐建议
"""

import argparse
import base64
import sys
from pathlib import Path
from typing import Any

import yaml
from openai import OpenAI

from base_reviewer import BaseReviewer

# ────────────────────────────────────────────────
# 配置加载
# ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _ROOT / "config" / "ollama_review.yaml"
_FALLBACK_CONFIG_PATH = _ROOT / "config" / "gemini_review.yaml"

DEFAULT_CONFIG: dict[str, Any] = {
    # 路径参数（相对路径默认基于项目根目录）
    "candidates": "data/candidates/candidates_latest.json",
    "kline_dir": "data/kline",
    "output_dir": "data/review",
    "prompt_path": "agent/prompt.md",
    # Ollama 参数
    "model": "qwen2.5vl:7b",
    "ollama_base_url": "http://localhost:11434/v1",
    "temperature": 0.2,
    "request_delay": 2,
    "skip_existing": False,
    "suggest_min_score": 4.0,
}


def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _ROOT) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    # 优先用专属配置，其次 fallback 到 gemini_review.yaml 的通用字段
    cfg_path = config_path or (
        _DEFAULT_CONFIG_PATH if _DEFAULT_CONFIG_PATH.exists() else _FALLBACK_CONFIG_PATH
    )
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = {**DEFAULT_CONFIG, **raw}

    # BaseReviewer 依赖这些路径字段为 Path 对象
    cfg["candidates"] = _resolve_cfg_path(cfg["candidates"])
    cfg["kline_dir"] = _resolve_cfg_path(cfg["kline_dir"])
    cfg["output_dir"] = _resolve_cfg_path(cfg["output_dir"])
    cfg["prompt_path"] = _resolve_cfg_path(cfg["prompt_path"])

    return cfg


# ────────────────────────────────────────────────
# OllamaReviewer
# ────────────────────────────────────────────────

class OllamaReviewer(BaseReviewer):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        base_url = config.get("ollama_base_url", "http://localhost:11434/v1")
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",   # Ollama 不校验 API Key，填任意字符串即可
        )
        self.model = config.get("model", "qwen2.5vl:7b")
        self.temperature = float(config.get("temperature", 0.2))
        print(f"[INFO] 使用本地模型：{self.model}  地址：{base_url}")

    @staticmethod
    def _image_to_data_url(path: Path) -> str:
        """将图片文件编码为 base64 data URL。"""
        suffix = path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
        mime_type = mime_map.get(suffix, "image/jpeg")
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime_type};base64,{data}"

    def review_stock(self, code: str, day_chart: Path, prompt: str) -> dict:
        """
        调用本地 Ollama API，对单支股票进行图表分析，返回解析后的 JSON 结果。
        """
        image_url = self._image_to_data_url(day_chart)

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"股票代码：{code}\n\n"
                                "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
                                "并严格按照要求输出 JSON。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                },
            ],
        )

        response_text = response.choices[0].message.content
        if not response_text:
            raise RuntimeError(f"模型返回空响应，无法解析 JSON（code={code}）")

        result = self.extract_json(response_text)
        result["code"] = code  # 附加股票代码便于追溯
        return result


def main():
    parser = argparse.ArgumentParser(description="Ollama 本地多模态图表复评")
    parser.add_argument(
        "--config",
        default=None,
        help=f"配置文件路径（默认 {_DEFAULT_CONFIG_PATH.name}，不存在则 fallback 到 gemini_review.yaml）",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    reviewer = OllamaReviewer(config)
    reviewer.run()


if __name__ == "__main__":
    main()
