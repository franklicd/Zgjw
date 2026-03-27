"""
monthly_backtest.py
~~~~~~~~~~~~~~~~~~~
月度回测脚本：测试选股模型的胜负率

功能：
    1. 从过去 N 个月每月 1 日作为选股日期
    2. 选出当时排名前 10 的股票
    3. 追踪这些股票在第 30 天的涨跌幅
    4. 输出详细清单和胜负率统计

用法：
    python -m backtest.monthly_backtest
    python -m backtest.monthly_backtest --months 12 --top-n 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

from backtest.data_loader import (
    get_price_on_date,
    load_kline_data
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CANDIDATES_DIR = DATA_DIR / "candidates"
REVIEW_DIR = DATA_DIR / "review"


@dataclass
class StockResult:
    """单只股票的回测结果"""
    code: str
    rank: int
    score: float
    pick_price: float
    target_price: Optional[float]
    change_pct: Optional[float]
    pick_date: str
    target_date: str
    status: str  # "win", "lose", "tie", "no_data"


@dataclass
class MonthlyResult:
    """单月回测结果"""
    pick_date: str
    target_date: str
    stocks: list[StockResult]
    win_count: int
    lose_count: int
    tie_count: int
    no_data_count: int
    avg_gain: float
    win_rate: float


def get_monthly_dates(months: int) -> list[tuple[str, str]]:
    """
    获取过去 N 个月的选股日期和目标日期

    Returns:
        list of (pick_date, target_date) tuples, format: YYYY-MM-DD
    """
    dates = []
    now = datetime.now()

    for i in range(months):
        # 计算前 i 个月的 1 日
        if now.month <= i:
            year = now.year - (i - now.month) // 12 - 1
            month = 12 + (now.month - i - 1) % 12 + 1
        else:
            year = now.year
            month = now.month - i

        # 每月 1 日作为选股日期
        pick_date = datetime(year, month, 1)

        # 第 30 天作为目标日期（选股日后的第 30 个交易日约等于 30 天后）
        target_date = pick_date + timedelta(days=30)

        dates.append((
            pick_date.strftime("%Y-%m-%d"),
            target_date.strftime("%Y-%m-%d")
        ))

    return dates


# 注意：以下函数已移至 backtest.data_loader 模块，使用导入的版本
# - load_kline_data
# - get_price_on_date


def load_candidates(pick_date: str) -> list[dict]:
    """
    加载指定日期的候选股票数据
    """
    # 尝试精确匹配日期
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

    return data.get('candidates', [])


def load_suggestion_scores(pick_date: str) -> dict[str, dict]:
    """
    加载指定日期的 AI 评分数据

    Returns:
        {code: {rank, score, ...}}
    """
    # 找对应日期的 review 目录
    review_dirs = [d for d in REVIEW_DIR.iterdir() if d.is_dir()]

    # 优先匹配精确日期
    target_dir = REVIEW_DIR / pick_date
    if not target_dir.exists():
        # 找最近的
        pick_dt = datetime.strptime(pick_date, "%Y-%m-%d")
        for i in range(10):
            check_date = (pick_dt - timedelta(days=i)).strftime("%Y-%m-%d")
            target_dir = REVIEW_DIR / check_date
            if target_dir.exists():
                break
        else:
            return {}

    suggestion_file = target_dir / "suggestion.json"
    if not suggestion_file.exists():
        return {}

    with open(suggestion_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    scores = {}
    for rec in data.get('recommendations', []):
        scores[rec['code']] = {
            'rank': rec.get('rank', 0),
            'score': rec.get('total_score', 0),
            'verdict': rec.get('verdict', ''),
            'signal_type': rec.get('signal_type', '')
        }

    return scores


def run_backtest(months: int = 12, top_n: int = 10) -> list[MonthlyResult]:
    """
    运行回测

    Args:
        months: 回测月数
        top_n: 每月选前 N 只股票

    Returns:
        list of MonthlyResult
    """
    print(f"\n{'='*60}")
    print(f"开始回测：过去{months}个月，每月选前{top_n}只股票")
    print(f"{'='*60}\n")

    monthly_dates = get_monthly_dates(months)
    results = []
    kline_cache = {}

    for pick_date, target_date in monthly_dates:
        print(f"\n[处理] 选股日期：{pick_date} -> 目标日期：{target_date}")

        # 1. 加载候选数据
        candidates = load_candidates(pick_date)
        if not candidates:
            print(f"  [跳过] 无候选数据")
            continue

        print(f"  候选股票数：{len(candidates)}")

        # 2. 加载 AI 评分
        scores = load_suggestion_scores(pick_date)

        # 3. 确定选股逻辑：
        # - 如果有 AI 评分，按评分排名选前 N
        # - 如果没有 AI 评分，按候选列表顺序（量化初选结果）选前 N

        if scores:
            # 按 AI 评分排序
            scored_candidates = [
                {**c, 'score': scores.get(c['code'], {}).get('score', 0),
                       'rank': scores.get(c['code'], {}).get('rank', 999)}
                for c in candidates
                if c['code'] in scores
            ]
            scored_candidates.sort(key=lambda x: x['rank'])
            selected = scored_candidates[:top_n]
            print(f"  使用 AI 评分排名，已评分：{len(scored_candidates)} 只")
        else:
            # 无 AI 评分，使用候选列表顺序（按 turnover_n 排序的结果）
            selected = candidates[:top_n]
            for i, c in enumerate(selected):
                c['rank'] = i + 1
                c['score'] = 0
            print(f"  无 AI 评分，使用量化初选排名")

        # 4. 计算每只股票的表现
        stock_results = []
        for stock in selected:
            code = stock['code']
            rank = stock.get('rank', 0)
            score = stock.get('score', 0)

            # 选股日价格
            pick_price = get_price_on_date(code, pick_date, kline_cache)
            if pick_price is None:
                print(f"    [无数据] {code} 在 {pick_date}")
                stock_results.append(StockResult(
                    code=code, rank=rank, score=score,
                    pick_price=0, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, status="no_data"
                ))
                continue

            # 目标日价格
            target_price = get_price_on_date(code, target_date, kline_cache)

            if target_price is None:
                stock_results.append(StockResult(
                    code=code, rank=rank, score=score,
                    pick_price=pick_price, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, status="no_data"
                ))
                continue

            # 计算涨跌幅
            change_pct = ((target_price - pick_price) / pick_price) * 100
            status = "win" if change_pct > 0 else ("lose" if change_pct < 0 else "tie")

            stock_results.append(StockResult(
                code=code, rank=rank, score=score,
                pick_price=pick_price, target_price=target_price,
                change_pct=change_pct, pick_date=pick_date,
                target_date=target_date, status=status
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


def print_report(results: list[MonthlyResult]) -> None:
    """打印详细回测报告"""
    print(f"\n{'='*80}")
    print("月度回测详细报告")
    print(f"{'='*80}")

    for mr in results:
        print(f"\n--- {mr.pick_date} (目标日：{mr.target_date}) ---")
        print(f"胜率：{mr.win_rate:.1f}% ({mr.win_count}胜/{mr.lose_count}负/{mr.tie_count}平)，"
              f"平均收益：{mr.avg_gain:.2f}%")
        print(f"{'排名':>4} {'代码':>8} {'分数':>6} {'选股价':>10} {'目标价':>10} {'涨跌%':>10} {'状态':>6}")
        print("-" * 70)

        for stock in mr.stocks:
            if stock.status == "no_data":
                print(f"{stock.rank:>4} {stock.code:>8} {stock.score:>6.1f} "
                      f"{stock.pick_price:>10.2f} {'N/A':>10} {'N/A':>10} 无数据")
            else:
                status_icon = "✓" if stock.status == "win" else ("×" if stock.status == "lose" else "=")
                print(f"{stock.rank:>4} {stock.code:>8} {stock.score:>6.1f} "
                      f"{stock.pick_price:>10.2f} {stock.target_price:>10.2f} "
                      f"{stock.change_pct:>+10.2f}% {status_icon}")

    # 总体统计
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")

    total_wins = sum(mr.win_count for mr in results)
    total_loses = sum(mr.lose_count for mr in results)
    total_ties = sum(mr.tie_count for mr in results)
    total_no_data = sum(mr.no_data_count for mr in results)

    all_stocks = [s for mr in results for s in mr.stocks if s.change_pct is not None]
    overall_avg_gain = sum(s.change_pct for s in all_stocks) / len(all_stocks) if all_stocks else 0

    total_valid = total_wins + total_loses + total_ties
    overall_win_rate = (total_wins / total_valid * 100) if total_valid > 0 else 0

    print(f"总选股数：{len(all_stocks)}")
    print(f"总胜场：{total_wins}, 总负场：{total_loses}, 总平场：{total_ties}, 无数据：{total_no_data}")
    print(f"总体胜率：{overall_win_rate:.1f}%")
    print(f"总体平均收益：{overall_avg_gain:.2f}%")


def save_report(results: list[MonthlyResult], output_file: Path) -> None:
    """保存回测报告为 JSON"""
    output_data = {
        'summary': {
            'months_tested': len(results),
            'total_wins': sum(mr.win_count for mr in results),
            'total_loses': sum(mr.lose_count for mr in results),
            'total_ties': sum(mr.tie_count for mr in results),
            'overall_win_rate': sum(mr.win_count for mr in results) /
                                sum(mr.win_count + mr.lose_count + mr.tie_count for mr in results) * 100
                                if any(mr.win_count + mr.lose_count + mr.tie_count > 0 for mr in results) else 0,
        },
        'monthly_results': [asdict(mr) for mr in results]
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description="月度选股回测脚本")
    parser.add_argument('--months', type=int, default=12, help='回测月数（默认 12）')
    parser.add_argument('--top-n', type=int, default=10, help='每月选股数量（默认 10）')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 文件路径')

    args = parser.parse_args()

    results = run_backtest(months=args.months, top_n=args.top_n)

    if not results:
        print("\n[错误] 回测未能生成任何结果，请检查数据文件是否存在")
        sys.exit(1)

    print_report(results)

    if args.output:
        save_report(results, Path(args.output))


if __name__ == "__main__":
    main()
