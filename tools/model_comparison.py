#!/usr/bin/env python3
"""
model_comparison.py
对比测试不同模型对同一股票的评分结果。
支持：
- 已有评分结果读取（Qwen/Doubao等）
- Ollama 本地模型评分（Llama3.2-vision 等）

针对 Mac M 系列芯片优化：
- 使用 Metal 加速（通过 Ollama）
- 批量处理优化

用法：
    python tools/model_comparison.py                           # 对比最新5只股票
    python tools/model_comparison.py --top-n 10               # 对比前10只
    python tools/model_comparison.py --date 2026-03-30        # 指定日期
    python tools/model_comparison.py --model llama3.2-vision:11b  # 指定模型
"""

import argparse
import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# 默认配置
DEFAULT_TOP_N = 5
DEFAULT_REVIEW_DIR = ROOT / "data" / "review"
DEFAULT_KLINE_DIR = ROOT / "data" / "kline"
DEFAULT_PROMPT_PATH = ROOT / "agent" / "prompt.md"
DEFAULT_OLLAMA_MODEL = "llama3.2-vision:11b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/v1"


def find_latest_review_date() -> Optional[str]:
    """查找最新的评分日期目录"""
    if not DEFAULT_REVIEW_DIR.exists():
        return None
    dates = [d.name for d in DEFAULT_REVIEW_DIR.iterdir() if d.is_dir() and d.name.startswith("2026")]
    if not dates:
        return None
    return sorted(dates, reverse=True)[0]


def load_existing_scores(review_date: str) -> dict[str, Any]:
    """加载指定日期的已有评分结果"""
    suggestion_file = DEFAULT_REVIEW_DIR / review_date / "suggestion.json"
    if not suggestion_file.exists():
        raise FileNotFoundError(f"找不到评分结果：{suggestion_file}")

    with open(suggestion_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def get_top_stocks(suggestion_data: dict, top_n: int) -> list[dict]:
    """获取评分最高的N只股票"""
    recommendations = suggestion_data.get("recommendations", [])
    return recommendations[:top_n]


def load_prompt() -> str:
    """加载评分 prompt"""
    return DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")


def image_to_base64(path: Path) -> tuple[str, str]:
    """将图片文件转为 base64 字符串及对应 mime_type"""
    suffix = path.suffix.lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime_type = mime_map.get(suffix, "image/jpeg")
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8"), mime_type


def find_chart_image(pick_date: str, code: str) -> Optional[Path]:
    """查找股票对应的K线图"""
    date_dir = DEFAULT_KLINE_DIR / pick_date
    if not date_dir.exists():
        return None

    # 尝试多种格式
    for ext in ["_day.jpg", "_day.png", ".jpg", ".png"]:
        chart_path = date_dir / f"{code}{ext}"
        if chart_path.exists():
            return chart_path
    return None


def call_ollama_vision(
    model: str,
    prompt: str,
    image_path: Path,
    stock_code: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """
    调用 Ollama 本地 vision 模型进行评分

    针对 Mac M 系列芯片优化：
    - Ollama 自动使用 Metal 加速
    - 适当的请求间隔避免资源竞争
    """
    import requests

    b64_data, mime_type = image_to_base64(image_path)

    user_text = (
        f"股票代码：{stock_code}\n\n"
        "以下是该股票的 **日线图**，请按照系统提示中的框架进行分析，"
        "并严格按照要求输出 JSON。"
    )

    messages = [
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
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }

    start_time = time.monotonic()
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600,  # 10分钟超时，针对大模型优化
            )
            break
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s
                print(f"    ⏳ 超时，{wait_time}秒后重试 ({attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Ollama 调用超时（{stock_code}）：{e}")

    response.raise_for_status()
    elapsed = time.monotonic() - start_time

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    # 解析 JSON
    parsed = extract_json(content)
    parsed["code"] = stock_code
    parsed["model"] = model
    parsed["elapsed_seconds"] = round(elapsed, 2)

    return parsed


def extract_json(text: str) -> dict:
    """从模型输出中提取 JSON

    支持两种格式：
    1. 标准 JSON（符合 prompt 要求的格式）
    2. 非标准格式（如 Llama 的长输出），尝试从中解析评分
    """
    # 尝试提取 code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if code_block:
        text = code_block.group(1)

    # 尝试查找标准 JSON 对象
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > 0:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # 非标准输出，尝试解析评分
    # 常见格式：trend_structure: X, price_position: X, volume_behavior: X, previous_abnormal_move: X
    # 或者：趋势X分，位置X分，量价X分，异动X分
    # 或者：总分：X

    result = {
        "trend_reasoning": "",
        "position_reasoning": "",
        "volume_reasoning": "",
        "abnormal_move_reasoning": "",
        "signal_reasoning": "",
        "scores": {
            "trend_structure": 3,  # 默认值
            "price_position": 3,
            "volume_behavior": 3,
            "previous_abnormal_move": 3,
        },
        "total_score": 0,
        "signal_type": "unknown",
        "verdict": "WATCH",
        "comment": "",
    }

    # 提取各项分数
    # 格式1: "趋势结构: X分" 或 "trend_structure: X"
    score_patterns = [
        (r"趋势结构[：:]\s*(\d+)", "trend_structure"),
        (r"trend_structure[：:]\s*(\d+)", "trend_structure"),
        (r"价格位置[：:]\s*(\d+)", "price_position"),
        (r"price_position[：:]\s*(\d+)", "price_position"),
        (r"量价行为[：:]\s*(\d+)", "volume_behavior"),
        (r"volume_behavior[：:]\s*(\d+)", "volume_behavior"),
        (r"前期建仓异动[：:]\s*(\d+)", "previous_abnormal_move"),
        (r"previous_abnormal_move[：:]\s*(\d+)", "previous_abnormal_move"),
    ]

    for pattern, key in score_patterns:
        match = re.search(pattern, text)
        if match:
            result["scores"][key] = int(match.group(1))

    # 提取总分
    total_match = re.search(r"总分[：:]\s*(\d+(?:\.\d+)?)", text)
    if total_match:
        result["total_score"] = float(total_match.group(1))
    else:
        # 如果没有总分，计算各项之和
        scores = result["scores"]
        result["total_score"] = (
            scores.get("trend_structure", 3) * 0.2 +
            scores.get("price_position", 3) * 0.2 +
            scores.get("volume_behavior", 3) * 0.3 +
            scores.get("previous_abnormal_move", 3) * 0.3
        )

    # 提取 verdict
    verdict_match = re.search(r"verdict[:\s]*(PASS|WATCH|FAIL)", text, re.IGNORECASE)
    if verdict_match:
        result["verdict"] = verdict_match.group(1).upper()

    # 提取 signal_type
    signal_match = re.search(r"signal_type[:\s]*(trend_start|rebound|distribution_risk)", text, re.IGNORECASE)
    if signal_match:
        result["signal_type"] = signal_match.group(1)

    # 如果仍然无法解析有效评分，抛出异常
    if result["total_score"] == 0:
        raise ValueError(f"无法解析模型输出中的评分信息:\n{text[:500]}")

    return result


def calculate_score_diff(existing: dict, ollama: dict) -> dict:
    """计算评分差异"""
    existing_scores = existing.get("scores", {})
    ollama_scores = ollama.get("scores", {})

    diff = {}
    for key in ["trend_structure", "price_position", "volume_behavior", "previous_abnormal_move"]:
        existing_val = existing_scores.get(key, 0)
        ollama_val = ollama_scores.get(key, 0)
        diff[key] = {
            "existing": existing_val,
            "ollama": ollama_val,
            "diff": ollama_val - existing_val,
        }

    existing_total = existing.get("total_score", 0)
    ollama_total = ollama.get("total_score", 0)

    return {
        "scores": diff,
        "total_score": {
            "existing": existing_total,
            "ollama": ollama_total,
            "diff": round(ollama_total - existing_total, 2),
        },
        "verdict": {
            "existing": existing.get("verdict", ""),
            "ollama": ollama.get("verdict", ""),
        },
        "signal_type": {
            "existing": existing.get("signal_type", ""),
            "ollama": ollama.get("signal_type", ""),
        },
    }


def format_comparison_table(comparisons: list[dict], existing_model: str = "Doubao") -> str:
    """格式化对比表格

    Args:
        comparisons: 对比结果列表
        existing_model: 已有评分使用的模型名称
    """
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append(f"{'股票代码':^8} | {'模型':^22} | {'总分':^6} | {' verdict ':^8} | {'趋势':^4} | {'位置':^4} | {'量价':^4} | {'异动':^4}")
    lines.append("-" * 100)

    for comp in comparisons:
        code = comp["code"]

        # 已有评分（ Doubao API）
        e = comp["existing"]
        lines.append(
            f"{code:^8} | {existing_model:^22} | {e.get('total_score', 0):^6.1f} | "
            f"{e.get('verdict', ''):^8} | "
            f"{e.get('scores', {}).get('trend_structure', 0):^4} | "
            f"{e.get('scores', {}).get('price_position', 0):^4} | "
            f"{e.get('scores', {}).get('volume_behavior', 0):^4} | "
            f"{e.get('scores', {}).get('previous_abnormal_move', 0):^4}"
        )

        # Ollama 评分
        o = comp["ollama"]
        lines.append(
            f"{'':^8} | {o.get('model', 'llama')[:22]:^22} | {o.get('total_score', 0):^6.1f} | "
            f"{o.get('verdict', ''):^8} | "
            f"{o.get('scores', {}).get('trend_structure', 0):^4} | "
            f"{o.get('scores', {}).get('price_position', 0):^4} | "
            f"{o.get('scores', {}).get('volume_behavior', 0):^4} | "
            f"{o.get('scores', {}).get('previous_abnormal_move', 0):^4}"
        )

        # 差异
        d = comp["diff"]["total_score"]["diff"]
        diff_str = f"+{d:.1f}" if d > 0 else f"{d:.1f}"
        lines.append(
            f"{'':^8} | {'差异':^22} | {diff_str:^6} | "
            f"{'':^8} | "
            f"{comp['diff']['scores'].get('trend_structure', {}).get('diff', 0):^+4} | "
            f"{comp['diff']['scores'].get('price_position', {}).get('diff', 0):^+4} | "
            f"{comp['diff']['scores'].get('volume_behavior', {}).get('diff', 0):^+4} | "
            f"{comp['diff']['scores'].get('previous_abnormal_move', {}).get('diff', 0):^+4}"
        )
        lines.append("-" * 100)

    return "\n".join(lines)


def run_comparison(
    top_n: int = DEFAULT_TOP_N,
    review_date: Optional[str] = None,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    existing_model: str = "Doubao",  # 已有评分使用的模型名称
    output_file: Optional[Path] = None,
):
    """运行对比测试

    Args:
        top_n: 对比 Top N 股票
        review_date: 评分日期（默认最新）
        ollama_model: Ollama 本地模型名称
        existing_model: 已有评分使用的模型名称（如 Doubao、Qwen 等）
        output_file: 输出文件路径
    """

    # 1. 查找最新评分日期
    if review_date is None:
        review_date = find_latest_review_date()
        if review_date is None:
            raise ValueError("找不到任何评分结果")

    print(f"[INFO] 使用评分日期：{review_date}")
    print(f"[INFO] 已有评分模型：{existing_model}")
    print(f"[INFO] 本地模型：{ollama_model}")

    # 2. 加载已有评分（Doubao API 评分结果）
    print(f"[INFO] 加载 {existing_model} API 评分结果...")
    suggestion_data = load_existing_scores(review_date)

    # 3. 获取 Top N 股票
    top_stocks = get_top_stocks(suggestion_data, top_n)
    if not top_stocks:
        raise ValueError(f"日期 {review_date} 没有推荐的股票")

    print(f"[INFO] 选取 Top {len(top_stocks)} 股票：{[s['code'] for s in top_stocks]}")

    # 4. 加载 prompt 和模型配置
    prompt = load_prompt()

    # 5. 加载每只股票的已有评分详情
    existing_details = {}
    for stock in top_stocks:
        code = stock["code"]
        json_file = DEFAULT_REVIEW_DIR / review_date / f"{code}.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                existing_details[code] = json.load(f)

    # 6. 对每只股票进行 Ollama 评分
    print(f"\n[INFO] 使用 Ollama 模型进行评分：{ollama_model}")
    print(f"[INFO] Mac M 优化：Ollama 自动使用 Metal 加速\n")

    comparisons = []
    total_start = time.monotonic()

    for i, stock in enumerate(top_stocks, 1):
        code = stock["code"]

        # 获取已有评分
        existing = existing_details.get(code, {})
        if not existing:
            # 从 suggestion 中获取基本信息
            existing = {
                "code": code,
                "total_score": stock.get("total_score", 0),
                "verdict": stock.get("verdict", ""),
                "signal_type": stock.get("signal_type", ""),
                "scores": {},  # suggestion 中没有详细分数
            }

        # 查找 K 线图
        # pick_date 在 candidates 中，需要找到对应日期的 K 线图
        # 从候选股票数据中获取日期
        candidates_file = ROOT / "data" / "candidates" / f"candidates_{review_date.replace('-', '-')}.json"
        if not candidates_file.exists():
            candidates_file = ROOT / "data" / "candidates" / "candidates_latest.json"

        pick_date = review_date
        with open(candidates_file, "r", encoding="utf-8") as f:
            cand_data = json.load(f)
            # 找到对应股票的日期
            for c in cand_data.get("candidates", []):
                if c["code"] == code:
                    pick_date = c.get("date", review_date)
                    break

        chart_path = find_chart_image(pick_date, code)

        if chart_path is None:
            print(f"[⚠️ ] {code} — 找不到 K 线图，跳过")
            continue

        print(f"[{i}/{len(top_stocks)}] 正在评分 {code}...")

        try:
            # 调用 Ollama
            ollama_result = call_ollama_vision(
                model=ollama_model,
                prompt=prompt,
                image_path=chart_path,
                stock_code=code,
            )

            # 计算差异
            diff = calculate_score_diff(existing, ollama_result)

            comparisons.append({
                "code": code,
                "existing": existing,
                "ollama": ollama_result,
                "diff": diff,
            })

            # 显示进度
            score_info = f"已有:{existing.get('total_score', '?')} -> Ollama:{ollama_result.get('total_score', '?')}"
            print(f"    ✓ {code} 完成 | {score_info} | 耗时:{ollama_result.get('elapsed_seconds', '?')}s")

        except Exception as e:
            print(f"[❌] {code} — 评分失败：{e}")
            continue

    total_elapsed = time.monotonic() - total_start

    # 7. 输出结果
    print(format_comparison_table(comparisons, existing_model=existing_model))

    # 8. 统计信息
    if comparisons:
        total_score_diffs = [c["diff"]["total_score"]["diff"] for c in comparisons]
        avg_diff = sum(total_score_diffs) / len(total_score_diffs)
        verdict_match = sum(
            1 for c in comparisons
            if c["diff"]["verdict"]["existing"] == c["diff"]["verdict"]["ollama"]
        )

        print(f"\n📊 统计信息：")
        print(f"   对比股票数：{len(comparisons)}")
        print(f"   总耗时：{total_elapsed:.1f} 秒")
        print(f"   平均评分差异：{avg_diff:+.2f}")
        print(f"   判定一致率：{verdict_match}/{len(comparisons)} ({verdict_match/len(comparisons)*100:.1f}%)")

    # 9. 保存结果
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = ROOT / "data" / "comparison" / f"comparison_{review_date}_{timestamp}.json"

    output_file.parent.mkdir(parents=True, exist_ok=True)

    result_data = {
        "review_date": review_date,
        "ollama_model": ollama_model,
        "comparison_date": datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "comparisons": comparisons,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n[💾] 对比结果已保存：{output_file}")

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="对比测试不同模型对股票的评分")
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"对比 Top N 股票（默认 {DEFAULT_TOP_N}）",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="指定评分日期（如 2026-03-30）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help=f"Ollama 本地模型名称（默认 {DEFAULT_OLLAMA_MODEL}）",
    )
    parser.add_argument(
        "--existing-model",
        type=str,
        default="Doubao",
        help="已有评分使用的模型名称（默认 Doubao）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    try:
        run_comparison(
            top_n=args.top_n,
            review_date=args.date,
            ollama_model=args.model,
            existing_model=args.existing_model,
            output_file=output_path,
        )
    except Exception as e:
        print(f"[❌] 对比测试失败：{e}")
        raise


if __name__ == "__main__":
    main()