"""
quant_backtest.py
~~~~~~~~~~~~~~~~~
快速回测脚本：使用量化初选排名（turnover_n 排序）

功能：
    1. 从过去 N 个月每月 1 日作为选股日期
    2. 选出量化初选排名前 10 的股票（按 turnover_n 排序）
    3. 追踪这些股票在第 30 天的涨跌幅
    4. 输出详细清单和胜负率统计

用法：
    python -m backtest.quant_backtest
    python -m backtest.quant_backtest --months 12 --top-n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

from backtest.data_loader import (
    get_trading_dates,
    find_first_trading_day,
    find_target_day,
    get_price_on_date,
    calc_max_gain,
    load_kline_data
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CANDIDATES_DIR = DATA_DIR / "candidates"

@dataclass
class StockResult:
    """单只股票的回测结果"""
    code: str
    name: str
    rank: int
    turnover_n: float
    pick_price: float
    target_price: Optional[float]
    change_pct: Optional[float]
    pick_date: str
    target_date: str
    status: str  # "win", "lose", "tie", "no_data"
    # 最大涨幅信息
    max_gain_pct: Optional[float] = None
    max_gain_day: Optional[int] = None


@dataclass
class MonthlyResult:
    """单月回测结果"""
    month: str
    pick_date: str
    target_date: str
    stocks: list[StockResult]
    win_count: int
    lose_count: int
    tie_count: int
    no_data_count: int
    avg_gain: float
    win_rate: float


# 注意：以下函数已移至 backtest.data_loader 模块，使用导入的版本
# - get_trading_dates
# - find_first_trading_day
# - find_target_day
# - get_price_on_date
# - load_kline_data
# - calc_max_gain


def load_candidates(pick_date: str) -> List[dict]:
    """加载指定日期的候选股票"""
    candidates_file = CANDIDATES_DIR / f"candidates_{pick_date}.json"

    if not candidates_file.exists():
        # 找最近的候选文件
        pick_dt = datetime.strptime(pick_date, "%Y-%m-%d")
        for i in range(10):
            check_date = (pick_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            candidates_file = CANDIDATES_DIR / f"candidates_{check_date}.json"
            if candidates_file.exists():
                pick_date = check_date
                break
        else:
            return []

    with open(candidates_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data.get('candidates', []), pick_date


def run_backtest(months: int = 12, top_n: int = 10) -> List[MonthlyResult]:
    """运行回测"""
    print(f"\n{'='*70}")
    print(f"量化回测：过去{months}个月，每月 turnover_n 前{top_n}只股票")
    print(f"{'='*70}\n")

    # 获取交易日列表
    print("[加载] 获取交易日列表...")
    trading_dates = get_trading_dates()
    if not trading_dates:
        print("[错误] 无法获取交易日列表")
        return []

    print(f"  共 {len(trading_dates)} 个交易日")
    print(f"  数据范围：{trading_dates[0]} 至 {trading_dates[-1]}")

    # 计算每个月的选股日期
    now = datetime.now()
    year = now.year
    month = now.month
    monthly_picks = []

    print(f"\n[计划] 回测 {months} 个月:")
    for i in range(months):
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1

        pick_date = find_first_trading_day(y, m, trading_dates)
        if pick_date:
            monthly_picks.append((f"{y:04d}-{m:02d}", pick_date))
            print(f"  {y:04d}-{m:02d}: 选股日={pick_date}")
        else:
            print(f"  {y:04d}-{m:02d}: 无数据")

    print(f"\n开始回测...\n")

    results = []
    kline_cache = {}

    for month_str, pick_date in monthly_picks:
        print(f"\n{'='*60}")
        print(f"[月份] {month_str} | 选股日：{pick_date}")
        print(f"{'='*60}")

        # 1. 加载候选数据
        candidates, actual_date = load_candidates(pick_date)
        if not candidates:
            print(f"  [跳过] 无候选数据")
            continue

        if actual_date != pick_date:
            print(f"  [使用] 候选文件日期：{actual_date}")

        print(f"  候选股票数：{len(candidates)}")

        # 2. 按 turnover_n 排序选前 N
        # candidates 已经按 turnover_n 降序排列
        selected = candidates[:top_n]
        print(f"  选股数量：{len(selected)}")

        # 3. 计算目标日期
        target_date = find_target_day(pick_date, trading_dates, 30)
        if not target_date:
            # 使用自然日 +30
            target_date = (datetime.strptime(pick_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"  目标日（+30）：{target_date}")

        # 4. 计算每只股票的表现
        stock_results = []

        for rank, c in enumerate(selected, 1):
            code = c['code']
            name = c.get('name', '')
            turnover_n = c.get('turnover_n', 0)

            # 选股日价格
            pick_price = c.get('close', 0)
            if pick_price <= 0:
                pick_price = get_price_on_date(code, pick_date, kline_cache)

            if pick_price is None or pick_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, turnover_n=turnover_n,
                    pick_price=0, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, status="no_data"
                ))
                continue

            # 目标日价格
            target_price = get_price_on_date(code, target_date, kline_cache)

            if target_price is None or target_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, turnover_n=turnover_n,
                    pick_price=pick_price, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, status="no_data"
                ))
                continue

            # 计算涨跌幅
            change_pct = ((target_price - pick_price) / pick_price) * 100
            status = "win" if change_pct > 0 else ("lose" if change_pct < 0 else "tie")

            # 计算期间最大涨幅
            max_gain_pct, max_gain_day = calc_max_gain(
                code, pick_date, target_date, pick_price, trading_dates, kline_cache)

            stock_results.append(StockResult(
                code=code, name=name, rank=rank, turnover_n=turnover_n,
                pick_price=pick_price, target_price=target_price,
                change_pct=change_pct, pick_date=pick_date,
                target_date=target_date, status=status,
                max_gain_pct=max_gain_pct, max_gain_day=max_gain_day
            ))

        # 5. 统计月度结果
        win_count = sum(1 for s in stock_results if s.status == "win")
        lose_count = sum(1 for s in stock_results if s.status == "lose")
        tie_count = sum(1 for s in stock_results if s.status == "tie")
        no_data_count = sum(1 for s in stock_results if s.status == "no_data")

        valid_results = [s for s in stock_results if s.change_pct is not None]
        avg_gain = sum(s.change_pct for s in valid_results) / len(valid_results) if valid_results else 0

        effective_total = win_count + lose_count + tie_count
        win_rate = (win_count / effective_total * 100) if effective_total > 0 else 0

        monthly_result = MonthlyResult(
            month=month_str,
            pick_date=pick_date,
            target_date=target_date,
            stocks=stock_results,
            win_count=win_count,
            lose_count=lose_count,
            tie_count=tie_count,
            no_data_count=no_data_count,
            avg_gain=avg_gain,
            win_rate=win_rate
        )
        results.append(monthly_result)

        print(f"  结果：胜={win_count}, 负={lose_count}, 平={tie_count}, 无数据={no_data_count}")
        print(f"  胜率：{win_rate:.1f}%, 平均收益：{avg_gain:.2f}%")

    return results


def print_report(results: List[MonthlyResult]) -> None:
    """打印详细回测报告"""
    print(f"\n{'='*110}")
    print("量化回测详细报告")
    print(f"{'='*110}")

    for mr in results:
        print(f"\n--- {mr.month} (选股日：{mr.pick_date}, 目标日：{mr.target_date}) ---")
        print(f"胜率：{mr.win_rate:.1f}% ({mr.win_count}胜/{mr.lose_count}负/{mr.tie_count}平)，"
              f"平均收益：{mr.avg_gain:.2f}%")
        print(f"{'排名':>4} {'代码':>8} {'名称':>10} {'选股价':>10} {'目标价':>10} {'涨跌%':>10} {'最大涨幅%':>12} {'第几日':>8}")
        print("-" * 110)

        for stock in mr.stocks:
            if stock.status == "no_data":
                print(f"{stock.rank:>4} {stock.code:>8} {stock.name:>10} "
                      f"{stock.pick_price:>10.2f} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
            else:
                status_icon = "+" if stock.status == "win" else ("-" if stock.status == "lose" else "0")
                max_gain_str = f"{stock.max_gain_pct:+.2f}" if stock.max_gain_pct is not None else "N/A"
                max_day_str = str(stock.max_gain_day) if stock.max_gain_day is not None else "N/A"
                print(f"{stock.rank:>4} {stock.code:>8} {stock.name:>10} "
                      f"{stock.pick_price:>10.2f} {stock.target_price:>10.2f} "
                      f"{stock.change_pct:>+10.2f}% {max_gain_str:>12} {max_day_str:>8} {status_icon}")

    # 总体统计
    print(f"\n{'='*110}")
    print("总体统计")
    print(f"{'='*110}")

    total_wins = sum(mr.win_count for mr in results)
    total_loses = sum(mr.lose_count for mr in results)
    total_ties = sum(mr.tie_count for mr in results)
    total_no_data = sum(mr.no_data_count for mr in results)

    all_stocks = [s for mr in results for s in mr.stocks if s.change_pct is not None]
    overall_avg_gain = sum(s.change_pct for s in all_stocks) / len(all_stocks) if all_stocks else 0

    total_valid = total_wins + total_loses + total_ties
    overall_win_rate = (total_wins / total_valid * 100) if total_valid > 0 else 0

    print(f"回测月数：{len(results)}")
    print(f"总选股数：{len(all_stocks)}")
    print(f"总胜场：{total_wins}, 总负场：{total_loses}, 总平场：{total_ties}, 无数据：{total_no_data}")
    print(f"总体胜率：{overall_win_rate:.1f}%")
    print(f"总体平均收益：{overall_avg_gain:.2f}%")

    # 月均统计
    if results:
        avg_monthly_win_rate = sum(mr.win_rate for mr in results) / len(results)
        avg_monthly_gain = sum(mr.avg_gain for mr in results) / len(results)
        print(f"\n月均胜率：{avg_monthly_win_rate:.1f}%")
        print(f"月均收益：{avg_monthly_gain:.2f}%")


def save_report(results: List[MonthlyResult], output_file: Path) -> None:
    """保存回测报告为 JSON"""
    output_data = {
        'summary': {
            'months_tested': len(results),
            'total_wins': sum(mr.win_count for mr in results),
            'total_loses': sum(mr.lose_count for mr in results),
            'total_ties': sum(mr.tie_count for mr in results),
            'overall_win_rate': (
                sum(mr.win_count for mr in results) /
                sum(mr.win_count + mr.lose_count + mr.tie_count for mr in results) * 100
                if any(mr.win_count + mr.lose_count + mr.tie_count > 0 for mr in results) else 0
            ),
            'overall_avg_gain': (
                sum(s.change_pct for mr in results for s in mr.stocks if s.change_pct is not None) /
                len([s for mr in results for s in mr.stocks if s.change_pct is not None])
                if any(s.change_pct is not None for mr in results for s in mr.stocks) else 0
            )
        },
        'monthly_results': [asdict(mr) for mr in results]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description="量化回测脚本（无需 AI 评分）")
    parser.add_argument('--months', type=int, default=12, help='回测月数（默认 12）')
    parser.add_argument('--top-n', type=int, default=10, help='每月选股数量（默认 10）')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 文件路径')

    args = parser.parse_args()

    results = run_backtest(months=args.months, top_n=args.top_n)

    if not results:
        print("\n[错误] 回测未能生成任何结果，请检查候选数据是否存在")
        sys.exit(1)

    print_report(results)

    if args.output:
        save_report(results, Path(args.output))


if __name__ == "__main__":
    main()
