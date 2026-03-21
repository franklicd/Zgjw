"""
scripts/export_kline_charts.py
AgentTrader · 批量导出候选股票 K线图（日线 + 周线）

用法：
    python scripts/export_kline_charts.py [--date YYYY-MM-DD] [--bars 120] [--weekly-bars 60]

输出目录：
    data/kline/<date>/<code>_day.jpg
    data/kline/<date>/<code>_week.jpg

依赖：
    pip install kaleido   （Plotly 静态图导出必需）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "dashboard"))

from components.charts import make_daily_chart, make_weekly_chart  # noqa: E402


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _load_candidates(candidates_path: Path) -> tuple[list[str], str]:
    """从 candidates JSON 文件中读取股票代码列表及 pick_date。

    Returns:
        (codes, pick_date)  pick_date 为空字符串时表示 JSON 中无该字段。
    """
    if not candidates_path.exists():
        print(f"[ERROR] 候选文件不存在：{candidates_path}")
        sys.exit(1)
    with open(candidates_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    codes = [c["code"] for c in data.get("candidates", [])]
    pick_date = data.get("pick_date", "")
    print(f"[INFO] 候选股票数量：{len(codes)}  pick_date：{pick_date or '(未设置)'}  来源：{candidates_path.name}")
    return codes, pick_date


def _load_raw(code: str, raw_dir: Path) -> pd.DataFrame:
    """加载单只股票日线 CSV。"""
    csv = raw_dir / f"{code}.csv"
    if not csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv)
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── 导出单张图 ────────────────────────────────────────────────────────────────

def _export_fig(fig, out_path: Path, width: int, height: int) -> None:
    """将 Plotly Figure 导出为 JPEG。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(
        str(out_path),
        format="jpg",
        width=width,
        height=height,
        scale=2,        # 2× 分辨率，适合屏幕阅读
    )


def _process_single_stock(args: Tuple[str, Path, Path, dict]) -> Tuple[str, bool, str]:
    """
    处理单只股票的图表生成任务

    Args:
        args: (code, raw_dir, out_root, config)

    Returns:
        (code, success, message)
    """
    code, raw_dir, out_root, config = args

    df_raw = _load_raw(code, raw_dir)
    if df_raw.empty:
        return code, False, "无日线数据"

    # ── 日线图 ────────────────────────────────────────────────────
    day_path = out_root / f"{code}_day.jpg"
    try:
        fig_day = make_daily_chart(
            df_raw, code,
            bars=config["bars"],
            height=config["day_height"],
        )
        _export_fig(fig_day, day_path, config["day_width"], config["day_height"])
        return code, True, f"→ {day_path.name}"
    except Exception as e:
        return code, False, f"日线导出失败：{e}"


# ── 主流程 ────────────────────────────────────────────────────────────────────

# 配置字典（直接修改此处）
CONFIG = {
    "candidates": str(_ROOT / "data" / "candidates" / "candidates_latest.json"),
    "raw_dir":    str(_ROOT / "data" / "raw"),
    "out_dir":    str(_ROOT / "data" / "kline"),
    "bars":       120,   # 日线显示 K 线数量（0 = 全部）
    "weekly_bars": 60,   # 周线显示 K 线数量（0 = 全部）
    "day_width":  1400,
    "day_height": 700,
    "week_width": 1400,
    "week_height": 700,
    "max_workers": 4,    # 并发线程数
}


def main() -> None:
    parser = argparse.ArgumentParser(description="批量导出候选股票K线图")
    parser.add_argument('--date', type=str, help='指定导出日期 (YYYY-MM-DD)，默认从candidates文件读取')
    parser.add_argument('--bars', type=int, help='日线显示K线数量（0 = 全部）')
    parser.add_argument('--weekly-bars', type=int, dest='weekly_bars', help='周线显示K线数量（0 = 全部）')
    parser.add_argument('--workers', type=int, help='并发线程数')
    parser.add_argument('--config', type=str, help='配置文件路径')

    args = parser.parse_args()

    candidates_path = Path(CONFIG["candidates"])
    raw_dir         = Path(CONFIG["raw_dir"])

    codes, pick_date = _load_candidates(candidates_path)

    # 使用命令行参数覆盖默认配置
    if args.date:
        pick_date = args.date
    if args.bars is not None:
        CONFIG["bars"] = args.bars
    if args.weekly_bars is not None:
        CONFIG["weekly_bars"] = args.weekly_bars
    if args.workers is not None:
        CONFIG["max_workers"] = args.workers

    # 导出日期直接读取 candidates.json 的 pick_date
    export_date = pick_date
    if not export_date:
        print("[ERROR] candidates.json 中未设置 pick_date，无法确定导出日期。")
        sys.exit(1)
    print(f"[INFO] 导出日期：{export_date}")

    out_root = Path(CONFIG["out_dir"]) / export_date

    ok_count    = 0
    skip_count  = 0

    # 准备任务参数
    tasks = [(code, raw_dir, out_root, CONFIG) for code in codes]

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        # 提交所有任务
        future_to_code = {executor.submit(_process_single_stock, task): task[0] for task in tasks}

        # 使用tqdm显示进度
        for future in tqdm(as_completed(future_to_code), total=len(tasks), desc="导出K线图", ncols=80):
            code, success, message = future.result()

            if success:
                print(f"[OK]   {code}  {message}")
                ok_count += 1
            else:
                print(f"[SKIP] {code}  — {message}")
                skip_count += 1

    print(
        f"\n导出完成：成功 {ok_count} 只，跳过 {skip_count} 只。"
        f"\n输出目录：{out_root}"
    )


if __name__ == "__main__":
    main()
