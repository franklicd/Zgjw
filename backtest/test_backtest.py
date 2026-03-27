"""
test_backtest.py
~~~~~~~~~~~~~~~~
快速测试回测脚本的核心功能（不运行完整流程）
"""
from __future__ import annotations

import sys
import csv
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

def get_trading_dates(code: str = "000001") -> list[str]:
    """获取所有交易日"""
    kline_file = RAW_DIR / f"{code}.csv"
    if not kline_file.exists():
        return []

    dates = []
    with open(kline_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dates.append(row[0])
    return sorted(dates)


def find_first_trading_day(year: int, month: int, trading_dates: list[str]) -> str:
    """找到指定月份第一个交易日"""
    target = f"{year:04d}-{month:02d}-01"
    for d in trading_dates:
        if d >= target and d.startswith(f"{year:04d}-{month:02d}"):
            return d
    return None


def find_target_day(pick_date: str, trading_dates: list[str], days: int = 30) -> str:
    """找到第 N 个交易日"""
    for i, d in enumerate(trading_dates):
        if d >= pick_date:
            if i + days < len(trading_dates):
                return trading_dates[i + days]
    return None


def main():
    print("测试回测核心功能...\n")

    # 1. 获取交易日
    print("[1] 获取交易日列表...")
    trading_dates = get_trading_dates()
    print(f"    共 {len(trading_dates)} 个交易日")
    print(f"    范围：{trading_dates[0]} 至 {trading_dates[-1]}")

    # 2. 计算过去 12 个月每月的第一个交易日
    print("\n[2] 计算每月第一个交易日:")
    now = datetime.now()
    monthly_dates = []

    # 从当前月份开始，往前推 N 个月
    year = now.year
    month = now.month

    for i in range(12):
        # 计算当前迭代的年月
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1

        pick_date = find_first_trading_day(y, m, trading_dates)
        if pick_date:
            monthly_dates.append(pick_date)
            target_date = find_target_day(pick_date, trading_dates, 30)
            print(f"    {y:04d}-{m:02d}: 选股日={pick_date}, 目标日={target_date or 'N/A'}")
        else:
            print(f"    {y:04d}-{m:02d}: 无数据")

    # 3. 测试数据加载
    print("\n[3] 测试 K 线数据加载:")
    test_code = "000001"
    kline_file = RAW_DIR / f"{test_code}.csv"
    if kline_file.exists():
        with open(kline_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            print(f"    {test_code}: {len(rows)} 条记录")
            if rows:
                first = rows[0]
                last = rows[-1]
                print(f"    最早：{first['date']}, 最新：{last['date']}")
    else:
        print(f"    {test_code}: 文件不存在")

    # 4. 测试候选数据
    print("\n[4] 候选数据文件:")
    candidates_dir = DATA_DIR / "candidates"
    if candidates_dir.exists():
        for f in sorted(candidates_dir.glob("*.json")):
            print(f"    {f.name}")
    else:
        print("    目录不存在")

    # 5. 测试 review 数据
    print("\n[5] AI 评分数据:")
    review_dir = DATA_DIR / "review"
    if review_dir.exists():
        dirs = [d.name for d in review_dir.iterdir() if d.is_dir()]
        for d in sorted(dirs):
            suggestion = review_dir / d / "suggestion.json"
            if suggestion.exists():
                print(f"    {d}: 有数据")
            else:
                print(f"    {d}: 无 suggestion.json")
    else:
        print("    目录不存在")

    print("\n✅ 测试完成!")
    print("\n提示：现在可以运行完整回测:")
    print("  python -m backtest.historical_backtest --months 12 --top-n 10")


if __name__ == "__main__":
    main()
