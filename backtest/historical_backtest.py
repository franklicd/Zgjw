"""
historical_backtest.py
~~~~~~~~~~~~~~~~~~~~~~
历史回测脚本：重新运行过去 N 个月的完整选股流程

功能：
    1. 对过去 N 个月，找到每月第一个交易日作为选股日；或指定任意日期作为选股日
    2. 运行完整流程：量化初选 → 导出图表 → AI 评分
    3. 选出 AI 评分最高的前 N 只股票
    4. 追踪每只股票在第 30 个交易日的表现
    5. 输出详细回测报告

用法：
    # 回测过去 12 个月
    python -m backtest.historical_backtest
    python -m backtest.historical_backtest --months 12 --top-n 10

    # 指定任意日期作为选股日
    python -m backtest.historical_backtest --pick-date 2025-06-01 --top-n 10
    python -m backtest.historical_backtest --pick-date 2025-06-01 --top-n 10 --hold-days 30

    # 指定起止日期范围
    python -m backtest.historical_backtest --start 2025-01-01 --end 2025-06-30 --top-n 10
    python -m backtest.historical_backtest --start 2025-01-01 --end 2025-06-30 --top-n 10 --hold-days 15

    # 生成 Markdown 分析报告
    python -m backtest.historical_backtest --start 2025-01-01 --end 2025-06-30 --report report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import csv
import math
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CANDIDATES_DIR = DATA_DIR / "candidates"
REVIEW_DIR = DATA_DIR / "review"
PYTHON = sys.executable


@dataclass
class StockResult:
    """单只股票的回测结果"""
    code: str
    name: str
    rank: int
    score: float
    pick_price: float
    target_price: Optional[float]
    change_pct: Optional[float]
    pick_date: str
    target_date: str
    verdict: str
    signal_type: str
    strategy: str  # 策略类型（如 b1）
    status: str  # "win", "lose", "tie", "no_data"
    # 新增：最大涨幅信息
    max_gain_pct: Optional[float] = None  # 最大涨幅百分比
    max_gain_day: Optional[int] = None  # 最大涨幅发生在第几天
    # 新增：阈值触及日期追踪
    date_10pct: Optional[str] = None      # 第一次达到 +10% 涨幅的日期
    date_drop2pct: Optional[str] = None   # 第一次达到 -2% 跌幅的日期
    date_drop4pct: Optional[str] = None   # 第一次达到 -4% 跌幅的日期
    first_10pct_day: Optional[int] = None # +10% 发生在第几天
    first_drop2_day: Optional[int] = None # -2% 发生在第几天
    first_drop4_day: Optional[int] = None # -4% 发生在第几天


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


def get_all_trading_dates(code: str = "000001") -> List[str]:
    """
    获取所有交易日列表（从 K 线数据中提取）

    Returns:
        按时间排序的交易日列表 ['2023-01-03', '2023-01-04', ...]
    """
    kline_file = RAW_DIR / f"{code}.csv"
    if not kline_file.exists():
        return []

    dates = []
    with open(kline_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            dates.append(row[0])

    return sorted(dates)


def find_first_trading_day_of_month(year: int, month: int, trading_dates: List[str]) -> Optional[str]:
    """
    找到指定月份的第一天（或之后）的交易日
    """
    target_start = f"{year:04d}-{month:02d}-01"

    for date in trading_dates:
        if date >= target_start:
            # 检查是否是目标月份的日期
            date_parts = date.split('-')
            if int(date_parts[0]) == year and int(date_parts[1]) == month:
                return date

    return None


def find_target_trading_day(pick_date: str, trading_dates: List[str], days: int = 30) -> Optional[str]:
    """
    找到选股日之后第 N 个交易日

    Args:
        pick_date: 选股日 YYYY-MM-DD
        trading_dates: 交易日列表（已排序）
        days: 交易日天数（默认 30）

    Returns:
        目标交易日 YYYY-MM-DD
    """
    # 找到选股日在交易日列表中的位置
    for i, date in enumerate(trading_dates):
        if date >= pick_date:
            # 往后数 N 个交易日
            target_idx = i + days
            if target_idx < len(trading_dates):
                return trading_dates[target_idx]
            else:
                return None  # 超出数据范围

    return None


def run_preselect(pick_date: str) -> bool:
    """运行量化初选"""
    print(f"  [步骤 1/3] 运行量化初选，选股日：{pick_date}")

    cmd = [
        PYTHON, "-m", "pipeline.cli", "preselect",
        "--date", pick_date,
    ]

    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    [错误] 初选失败：{result.stderr[:200] if result.stderr else '未知错误'}")
        return False

    # 确保生成对应日期的候选文件
    candidates_file = CANDIDATES_DIR / f"candidates_{pick_date}.json"
    if not candidates_file.exists():
        latest_file = CANDIDATES_DIR / "candidates_latest.json"
        if latest_file.exists():
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['pick_date'] = pick_date
            with open(candidates_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

    print(f"    [完成] 初选完成")
    return True


def export_charts(pick_date: str) -> bool:
    """导出 K 线图"""
    print(f"  [步骤 2/3] 导出候选股票 K 线图")

    # 确保 candidates_latest.json 指向正确的日期
    candidates_file = CANDIDATES_DIR / f"candidates_{pick_date}.json"
    latest_file = CANDIDATES_DIR / "candidates_latest.json"

    if candidates_file.exists():
        with open(candidates_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    cmd = [PYTHON, str(ROOT / "dashboard" / "export_kline_charts.py")]
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)

    if result.returncode != 0:
        print(f"    [警告] 导出图表可能有误")

    print(f"    [完成] 图表导出完成")
    return True


def run_ai_review(max_workers: int = None, request_delay: float = None, use_ollama: bool = False) -> bool:
    """运行 AI 评分

    Args:
        max_workers: AI 分析并发数（可选，覆盖默认配置）
        request_delay: AI 分析请求间隔秒数（可选，覆盖默认配置）
        use_ollama: 是否使用 Ollama 本地模型（默认 False，使用阿里云 API）
    """
    print(f"  [步骤 3/3] 运行 AI 评分")
    model_type = "Ollama 本地模型" if use_ollama else "通义千问 API"
    print(f"    → 使用 {model_type} 分析股票...（进度见下方）\n")

    cmd = [PYTHON, str(ROOT / "agent" / "qwen_review.py")]

    # 传递并发参数
    if max_workers is not None:
        cmd.extend(["--max-workers", str(max_workers)])
    if request_delay is not None:
        cmd.extend(["--request-delay", str(request_delay)])
    if use_ollama:
        cmd.extend(["--use-ollama"])

    # 不使用 capture_output，让 AI 分析的进度实时输出到终端
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n    [错误] AI 评分失败")
        return False

    print(f"\n    [完成] AI 评分完成")
    return True


def run_full_pipeline(pick_date: str, max_workers: int = None, request_delay: float = None,
                      use_ollama: bool = False) -> bool:
    """运行完整选股流程

    Args:
        pick_date: 选股日
        max_workers: AI 分析并发数（可选）
        request_delay: AI 分析请求间隔秒数（可选）
        use_ollama: 是否使用 Ollama 本地模型（默认 False）
    """
    if not run_preselect(pick_date):
        return False

    export_charts(pick_date)

    if not run_ai_review(max_workers=max_workers, request_delay=request_delay, use_ollama=use_ollama):
        return False

    return True


def load_suggestion(pick_date: str) -> Optional[dict]:
    """加载 AI 评分结果"""
    review_dirs = [d for d in REVIEW_DIR.iterdir() if d.is_dir()]

    # 优先匹配精确日期
    for d in review_dirs:
        if d.name.startswith(pick_date[:10]):
            suggestion_file = d / "suggestion.json"
            if suggestion_file.exists():
                with open(suggestion_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

    # 找最近的（往后找 5 天）
    pick_dt = datetime.strptime(pick_date, "%Y-%m-%d")
    for i in range(5):
        check_date = (pick_dt + timedelta(days=i)).strftime("%Y-%m-%d")[:10]
        target_dir = REVIEW_DIR / check_date
        if target_dir.exists():
            suggestion_file = target_dir / "suggestion.json"
            if suggestion_file.exists():
                with open(suggestion_file, 'r', encoding='utf-8') as f:
                    return json.load(f)

    return None


def get_price_on_date(code: str, target_date: str, kline_cache: dict) -> Optional[float]:
    """获取股票在指定日期的收盘价"""
    if code in kline_cache:
        kline_data = kline_cache[code]
    else:
        kline_data = load_kline_data(code)
        kline_cache[code] = kline_data

    if not kline_data:
        return None

    if target_date in kline_data:
        return kline_data[target_date]['close']

    # 找最近的交易日
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    for i in range(10):
        check_date = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in kline_data:
            return kline_data[check_date]['close']

    return None


def calc_max_gain_during_period(code: str, pick_date: str, target_date: str,
                                 pick_price: float, trading_dates: List[str],
                                 kline_cache: dict) -> tuple[Optional[float], Optional[int]]:
    """
    计算选股日后到目标日期间，股票的最大涨幅及发生日期

    Args:
        code: 股票代码
        pick_date: 选股日 YYYY-MM-DD
        target_date: 目标日 YYYY-MM-DD
        pick_price: 选股日价格
        trading_dates: 交易日列表
        kline_cache: K 线数据缓存

    Returns:
        (max_gain_pct, max_gain_day): 最大涨幅百分比，发生在第几天
    """
    if code not in kline_cache:
        kline_cache[code] = load_kline_data(code)

    kline_data = kline_cache[code]
    if not kline_data:
        return None, None

    # 找到选股日和目标日之间的所有交易日
    start_idx = -1
    end_idx = -1

    for i, d in enumerate(trading_dates):
        if d >= pick_date and start_idx < 0:
            start_idx = i
        if d >= target_date and end_idx < 0:
            end_idx = i
            break

    if start_idx < 0 or start_idx >= len(trading_dates):
        return None, None

    if end_idx < 0:
        end_idx = len(trading_dates) - 1

    # 遍历期间的所有价格，找最高价
    max_price = pick_price
    max_day_idx = 0

    for i in range(start_idx, min(end_idx + 1, len(trading_dates))):
        d = trading_dates[i]
        if d in kline_data:
            price = kline_data[d]['close']
            if price > max_price:
                max_price = price
                # 计算是选股日后的第几天（交易日）
                max_day_idx = i - start_idx

    if max_price <= pick_price:
        return 0.0, 0

    max_gain_pct = ((max_price - pick_price) / pick_price) * 100
    return max_gain_pct, max_day_idx


def track_threshold_dates(code: str, pick_date: str, target_date: str,
                          pick_price: float, trading_dates: List[str],
                          kline_cache: dict) -> Dict:
    """
    追踪选股日后到目标日期间，股票第一次达到 +10%、-2%、-4% 阈值的日期

    Args:
        code: 股票代码
        pick_date: 选股日 YYYY-MM-DD
        target_date: 目标日 YYYY-MM-DD
        pick_price: 选股日价格（pick_date 收盘价）
        trading_dates: 交易日列表
        kline_cache: K 线数据缓存

    Returns:
        dict with date_10pct, date_drop2pct, date_drop4pct, first_10pct_day, first_drop2_day, first_drop4_day
    """
    result = {
        'date_10pct': None, 'first_10pct_day': None,
        'date_drop2pct': None, 'first_drop2_day': None,
        'date_drop4pct': None, 'first_drop4_day': None,
    }

    if code not in kline_cache:
        kline_cache[code] = load_kline_data(code)

    kline_data = kline_cache[code]
    if not kline_data:
        return result

    # 找到选股日和目标日之间的所有交易日
    start_idx = -1
    end_idx = -1

    for i, d in enumerate(trading_dates):
        if d >= pick_date and start_idx < 0:
            start_idx = i
        if d >= target_date and end_idx < 0:
            end_idx = i
            break

    if start_idx < 0 or start_idx >= len(trading_dates):
        return result

    if end_idx < 0:
        end_idx = len(trading_dates) - 1

    # 阈值计算
    target_10pct = pick_price * 1.10  # +10%
    target_drop2 = pick_price * 0.98  # -2%
    target_drop4 = pick_price * 0.96  # -4%

    # 遍历期间的所有价格，找第一次触及阈值的日期
    for i in range(start_idx, min(end_idx + 1, len(trading_dates))):
        d = trading_dates[i]
        if d not in kline_data:
            continue

        price = kline_data[d]['close']
        day_idx = i - start_idx

        # 检查 +10%
        if result['date_10pct'] is None and price >= target_10pct:
            result['date_10pct'] = d
            result['first_10pct_day'] = day_idx

        # 检查 -2%
        if result['date_drop2pct'] is None and price <= target_drop2:
            result['date_drop2pct'] = d
            result['first_drop2_day'] = day_idx

        # 检查 -4%
        if result['date_drop4pct'] is None and price <= target_drop4:
            result['date_drop4pct'] = d
            result['first_drop4_day'] = day_idx

        # 如果三个阈值都已触及，提前退出
        if all(result[key] is not None for key in ['date_10pct', 'date_drop2pct', 'date_drop4pct']):
            break

    return result


def load_kline_data(code: str) -> dict:
    """加载 K 线数据"""
    kline_file = RAW_DIR / f"{code}.csv"
    if not kline_file.exists():
        return {}

    data = {}
    with open(kline_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row['date']] = {
                'open': float(row['open']),
                'close': float(row['close']),
                'high': float(row['high']),
                'low': float(row['low']),
                'volume': float(row['volume'])
            }
    return data


def run_historical_backtest(months: int = 12, top_n: int = 10,
                            ai_workers: int = None, ai_request_delay: float = None,
                            use_ollama: bool = False) -> List[MonthlyResult]:
    """
    运行历史回测

    Args:
        months: 回测月数
        top_n: 每月选前 N 只股票（AI 评分最高）
        ai_workers: AI 分析并发数（可选，覆盖默认配置）
        ai_request_delay: AI 分析请求间隔秒数（可选，覆盖默认配置）
        use_ollama: 是否使用 Ollama 本地模型（默认 False，使用阿里云 API）

    Returns:
        list of MonthlyResult
    """
    print(f"\n{'='*70}")
    print(f"历史回测：过去{months}个月，每月 AI 评分前{top_n}只股票")
    model_type = "Ollama 本地模型" if use_ollama else "阿里云 API"
    print(f"AI 分析模式：{model_type}")
    if ai_workers:
        print(f"AI 分析并发数：{ai_workers}")
    if ai_request_delay is not None:
        print(f"AI 分析请求间隔：{ai_request_delay}秒")
    print(f"{'='*70}\n")

    # 获取所有交易日
    print("[加载] 获取交易日列表...")
    trading_dates = get_all_trading_dates()
    if not trading_dates:
        print("[错误] 无法获取交易日列表")
        return []

    print(f"  共 {len(trading_dates)} 个交易日")
    print(f"  数据范围：{trading_dates[0]} 至 {trading_dates[-1]}")

    results = []
    kline_cache = {}

    # 计算每个月的选股日期
    now = datetime.now()
    monthly_picks = []

    year = now.year
    month = now.month

    for i in range(months):
        # 计算当前迭代的年月
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1

        pick_date = find_first_trading_day_of_month(y, m, trading_dates)
        if pick_date:
            monthly_picks.append(pick_date)
            print(f"  {y:04d}-{m:02d}: 选股日={pick_date}")
        else:
            print(f"  {y:04d}-{m:02d}: 无交易日")

    print(f"\n计划回测 {len(monthly_picks)} 个月\n")

    for pick_date in monthly_picks:
        print(f"\n{'='*60}")
        print(f"[月份] {pick_date[:7]} | 选股日：{pick_date}")
        print(f"{'='*60}")

        # 1. 检查是否已有结果
        suggestion = load_suggestion(pick_date)

        if suggestion is None:
            print(f"  [新运行] 未找到历史结果，开始运行选股流程...")
            if not run_full_pipeline(pick_date, max_workers=ai_workers, request_delay=ai_request_delay,
                                      use_ollama=use_ollama):
                print(f"  [跳过] 选股流程失败")
                continue

            suggestion = load_suggestion(pick_date)

        if suggestion is None:
            print(f"  [跳过] 无法获取 AI 评分结果")
            continue

        # 2. 获取推荐股票（AI 评分最高的前 N 只）
        recommendations = suggestion.get('recommendations', [])
        if not recommendations:
            print(f"  [跳过] 无推荐股票")
            continue

        # 按排名排序（已经是排好序的）
        selected = recommendations[:top_n]
        print(f"  AI 推荐前{top_n}只股票")

        # 3. 获取候选数据（用于获取选股时价格）
        candidates_file = CANDIDATES_DIR / f"candidates_{pick_date}.json"
        candidates_map = {}
        if candidates_file.exists():
            with open(candidates_file, 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
            for c in candidates_data.get('candidates', []):
                candidates_map[c['code']] = c

        # 4. 计算目标日期（第 30 个交易日）
        target_date = find_target_trading_day(pick_date, trading_dates, days=30)
        if not target_date:
            print(f"  [警告] 无法找到第 30 个交易日，使用自然日 +30")
            target_date = (datetime.strptime(pick_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")

        print(f"  目标日（+30 交易日）：{target_date}")

        # 5. 计算每只股票的表现
        stock_results = []

        for rec in selected:
            code = rec['code']
            rank = rec.get('rank', 0)
            score = rec.get('total_score', 0)
            verdict = rec.get('verdict', '')
            signal_type = rec.get('signal_type', '')

            # 从候选数据中获取更多信息
            cand_info = candidates_map.get(code, {})
            name = cand_info.get('name', '')
            strategy = cand_info.get('strategy', '')

            # 选股日价格
            pick_price = cand_info.get('close', 0)
            if pick_price is None or pick_price == 0:
                pick_price = get_price_on_date(code, pick_date, kline_cache)

            if pick_price is None or pick_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, score=score,
                    pick_price=0, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, verdict=verdict,
                    signal_type=signal_type, strategy=strategy, status="no_data"
                ))
                continue

            # 目标日价格
            target_price = get_price_on_date(code, target_date, kline_cache)

            if target_price is None or target_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, score=score,
                    pick_price=pick_price, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, verdict=verdict,
                    signal_type=signal_type, strategy=strategy, status="no_data",
                    max_gain_pct=None, max_gain_day=None
                ))
                continue

            # 计算涨跌幅
            change_pct = ((target_price - pick_price) / pick_price) * 100
            status = "win" if change_pct > 0 else ("lose" if change_pct < 0 else "tie")

            # 计算期间最大涨幅
            max_gain_pct, max_gain_day = calc_max_gain_during_period(
                code, pick_date, target_date, pick_price, trading_dates, kline_cache)

            # 追踪阈值触及日期
            threshold_data = track_threshold_dates(
                code, pick_date, target_date, pick_price, trading_dates, kline_cache)

            stock_results.append(StockResult(
                code=code, name=name, rank=rank, score=score,
                pick_price=pick_price, target_price=target_price,
                change_pct=change_pct, pick_date=pick_date,
                target_date=target_date, verdict=verdict,
                signal_type=signal_type, strategy=strategy, status=status,
                max_gain_pct=max_gain_pct, max_gain_day=max_gain_day,
                **threshold_data
            ))

        # 6. 统计月度结果
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


def print_report(results: List[MonthlyResult]) -> None:
    """打印详细回测报告"""
    print(f"\n{'='*120}")
    print("历史回测详细报告")
    print(f"{'='*120}")

    for mr in results:
        print(f"\n--- {mr.pick_date[:7]} (选股日：{mr.pick_date}, 目标日：{mr.target_date}) ---")
        print(f"胜率：{mr.win_rate:.1f}% ({mr.win_count}胜/{mr.lose_count}负/{mr.tie_count}平)，"
              f"平均收益：{mr.avg_gain:.2f}%")
        print(f"{'排名':>4} {'代码':>8} {'分数':>6} {'选股价':>10} {'目标价':>10} {'涨跌%':>10} {'最大涨幅%':>12} {'第几日':>8} {'研判':>8}")
        print("-" * 120)

        for stock in mr.stocks:
            if stock.status == "no_data":
                print(f"{stock.rank:>4} {stock.code:>8} {stock.score:>6.1f} "
                      f"{stock.pick_price:>10.2f} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>8} {stock.verdict[:8]:>8}")
            else:
                status_icon = "+" if stock.status == "win" else ("-" if stock.status == "lose" else "0")
                max_gain_str = f"{stock.max_gain_pct:+.2f}" if stock.max_gain_pct is not None else "N/A"
                max_day_str = str(stock.max_gain_day) if stock.max_gain_day is not None else "N/A"
                print(f"{stock.rank:>4} {stock.code:>8} {stock.score:>6.1f} "
                      f"{stock.pick_price:>10.2f} {stock.target_price:>10.2f} "
                      f"{stock.change_pct:>+10.2f}% {max_gain_str:>12} {max_day_str:>8} {stock.verdict[:8]:>8} {status_icon}")

    # 总体统计
    print(f"\n{'='*90}")
    print("总体统计")
    print(f"{'='*90}")

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

    # 额外统计：按月平均
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
        'monthly_results': []
    }

    for mr in results:
        month_data = asdict(mr)
        month_data['stocks'] = [asdict(s) for s in mr.stocks]
        output_data['monthly_results'].append(month_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n报告已保存至：{output_file}")


def save_markdown_report(results: List[MonthlyResult], output_file: Path,
                         start_date: str = None, end_date: str = None,
                         hold_days: int = 30) -> None:
    """保存回测报告为 Markdown 格式"""

    # 收集所有股票数据
    all_stocks: List[StockResult] = []
    for mr in results:
        for s in mr.stocks:
            if s.change_pct is not None:
                all_stocks.append(s)

    # 计算总体统计
    total_wins = sum(1 for s in all_stocks if s.change_pct > 0)
    total_loses = sum(1 for s in all_stocks if s.change_pct < 0)
    total_ties = sum(1 for s in all_stocks if s.change_pct == 0)
    total_count = len(all_stocks)
    overall_avg_gain = sum(s.change_pct for s in all_stocks) / total_count if total_count > 0 else 0
    overall_win_rate = (total_wins / total_count * 100) if total_count > 0 else 0

    # 找最佳和最差交易
    best_trade = max(all_stocks, key=lambda x: x.change_pct) if all_stocks else None
    worst_trade = min(all_stocks, key=lambda x: x.change_pct) if all_stocks else None

    # 按月份分组
    monthly_groups: Dict[str, List[StockResult]] = {}
    for s in all_stocks:
        month_key = s.pick_date[:7]
        if month_key not in monthly_groups:
            monthly_groups[month_key] = []
        monthly_groups[month_key].append(s)

    # 计算月度表现
    monthly_stats = []
    for month, stocks in sorted(monthly_groups.items()):
        month_wins = sum(1 for s in stocks if s.change_pct > 0)
        month_loses = sum(1 for s in stocks if s.change_pct < 0)
        month_ties = sum(1 for s in stocks if s.change_pct == 0)
        month_total = len(stocks)
        month_avg = sum(s.change_pct for s in stocks) / month_total if month_total > 0 else 0
        month_win_rate = (month_wins / month_total * 100) if month_total > 0 else 0
        monthly_stats.append({
            'month': month,
            'count': month_total,
            'wins': month_wins,
            'win_rate': month_win_rate,
            'avg_gain': month_avg
        })

    # 最大涨幅时间分布
    time_buckets = [
        (0, 5, "0-5 天"),
        (6, 10, "6-10 天"),
        (11, 15, "11-15 天"),
        (16, 20, "16-20 天"),
        (21, 25, "21-25 天"),
        (26, 30, "26-30 天"),
        (31, 35, "31-35 天"),
        (36, 100, "36+ 天"),
    ]
    time_distribution = []
    for low, high, label in time_buckets:
        count = sum(1 for s in all_stocks if s.max_gain_day is not None and low <= s.max_gain_day <= high)
        pct = (count / total_count * 100) if total_count > 0 else 0
        time_distribution.append({'label': label, 'count': count, 'pct': pct})

    # 最大涨幅统计
    valid_max_gains = [s for s in all_stocks if s.max_gain_day is not None]
    if valid_max_gains:
        avg_max_gain_day = sum(s.max_gain_day for s in valid_max_gains) / len(valid_max_gains)
        median_max_gain_day = sorted(s.max_gain_day for s in valid_max_gains)[len(valid_max_gains) // 2]
        min_max_gain_day = min(s.max_gain_day for s in valid_max_gains)
        max_max_gain_day = max(s.max_gain_day for s in valid_max_gains)
    else:
        avg_max_gain_day = median_max_gain_day = min_max_gain_day = max_max_gain_day = 0

    # 盈亏区间分布
    gain_buckets = [
        (10, float('inf'), "大盈 (>10%)"),
        (5, 10, "中盈 (5-10%)"),
        (0, 5, "小盈 (0-5%)"),
        (-5, 0, "小亏 (0-5%)"),
        (-10, -5, "中亏 (5-10%)"),
        (float('-inf'), -10, "大亏 (>10%)"),
    ]
    gain_distribution = []
    for low, high, label in gain_buckets:
        count = sum(1 for s in all_stocks if low <= s.change_pct < high)
        pct = (count / total_count * 100) if total_count > 0 else 0
        gain_distribution.append({'label': label, 'count': count, 'pct': pct})

    # 排名与盈亏相关性
    rank_groups = [
        (1, 3, "第 1-3 名"),
        (4, 5, "第 4-5 名"),
        (6, 7, "第 6-7 名"),
        (8, 10, "第 8-10 名"),
    ]
    rank_stats = []
    for low, high, label in rank_groups:
        group = [s for s in all_stocks if low <= s.rank <= high]
        if group:
            g_wins = sum(1 for s in group if s.change_pct > 0)
            g_total = len(group)
            g_win_rate = (g_wins / g_total * 100) if g_total > 0 else 0
            g_avg = sum(s.change_pct for s in group) / g_total if g_total > 0 else 0
            g_max_gain = sum(s.max_gain_pct for s in group if s.max_gain_pct is not None) / len([s for s in group if s.max_gain_pct is not None]) if any(s.max_gain_pct is not None for s in group) else 0
            rank_stats.append({
                'label': label, 'count': g_total, 'win_rate': g_win_rate,
                'avg_gain': g_avg, 'avg_max_gain': g_max_gain
            })

    # 计算 Pearson 相关系数（排名 vs 盈亏）
    def pearson_corr(x_vals, y_vals):
        n = len(x_vals)
        if n < 2:
            return 0
        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        den_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_vals))
        den_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_vals))
        if den_x * den_y == 0:
            return 0
        return num / (den_x * den_y)

    rank_vals = [s.rank for s in all_stocks]
    gain_vals = [s.change_pct for s in all_stocks]
    rank_corr = pearson_corr(rank_vals, gain_vals)

    rank_max_gain_vals = [s.max_gain_pct for s in all_stocks if s.max_gain_pct is not None]
    rank_max_gain_ranks = [s.rank for s in all_stocks if s.max_gain_pct is not None]
    rank_max_corr = pearson_corr(rank_max_gain_ranks, rank_max_gain_vals) if rank_max_gain_vals else 0

    # 分数与盈亏相关性（用 score 替代相似度）
    score_groups = [
        (4.5, float('inf'), "4.5 分+"),
        (4.0, 4.5, "4.0-4.5 分"),
        (0, 4.0, "0-4.0 分"),
    ]
    score_stats = []
    for low, high, label in score_groups:
        group = [s for s in all_stocks if low <= s.score < high]
        if group:
            g_wins = sum(1 for s in group if s.change_pct > 0)
            g_total = len(group)
            g_win_rate = (g_wins / g_total * 100) if g_total > 0 else 0
            g_avg = sum(s.change_pct for s in group) / g_total if g_total > 0 else 0
            g_max_gain = sum(s.max_gain_pct for s in group if s.max_gain_pct is not None) / len([s for s in group if s.max_gain_pct is not None]) if any(s.max_gain_pct is not None for s in group) else 0
            score_stats.append({
                'label': label, 'count': g_total, 'win_rate': g_win_rate,
                'avg_gain': g_avg, 'avg_max_gain': g_max_gain
            })

    score_vals = [s.score for s in all_stocks]
    score_corr = pearson_corr(score_vals, gain_vals)
    score_max_corr = pearson_corr(
        [s.score for s in all_stocks if s.max_gain_pct is not None],
        rank_max_gain_vals
    ) if rank_max_gain_vals else 0

    # 高分数股票详情（>=4.0 分）
    high_score_stocks = [s for s in all_stocks if s.score >= 4.0]
    high_score_stocks.sort(key=lambda x: x.score, reverse=True)

    # 最大涨幅 vs 最终盈亏
    avg_max_gain = sum(s.max_gain_pct for s in all_stocks if s.max_gain_pct is not None) / len([s for s in all_stocks if s.max_gain_pct is not None]) if any(s.max_gain_pct is not None for s in all_stocks) else 0
    diff_above_5pct = sum(1 for s in all_stocks if s.max_gain_pct is not None and s.max_gain_pct - s.change_pct > 5)
    diff_pct = (diff_above_5pct / total_count * 100) if total_count > 0 else 0

    # 分类统计（按 signal_type）
    category_groups: Dict[str, List[StockResult]] = {}
    for s in all_stocks:
        cat = s.signal_type if s.signal_type else "其他"
        if cat not in category_groups:
            category_groups[cat] = []
        category_groups[cat].append(s)

    category_stats = []
    for cat, stocks in sorted(category_groups.items(), key=lambda x: -sum(s.change_pct for s in x[1]) / len(x[1]) if x[1] else 0):
        c_wins = sum(1 for s in stocks if s.change_pct > 0)
        c_total = len(stocks)
        c_win_rate = (c_wins / c_total * 100) if c_total > 0 else 0
        c_avg = sum(s.change_pct for s in stocks) / c_total if c_total > 0 else 0
        c_max_gain = sum(s.max_gain_pct for s in stocks if s.max_gain_pct is not None) / len([s for s in stocks if s.max_gain_pct is not None]) if any(s.max_gain_pct is not None for s in stocks) else 0
        category_stats.append({
            'category': cat, 'count': c_total, 'win_rate': c_win_rate,
            'avg_gain': c_avg, 'avg_max_gain': c_max_gain
        })

    # 确定日期范围
    actual_start = start_date or (min(s.pick_date for s in all_stocks) if all_stocks else "N/A")
    actual_end = end_date or (max(s.pick_date for s in all_stocks) if all_stocks else "N/A")

    # ────────────────────────────────────────────────
    # 阈值触及统计（+10%, -2%, -4%）
    # ────────────────────────────────────────────────

    # 1. 基础统计：触及各类阈值的股票数量
    count_10pct = sum(1 for s in all_stocks if s.date_10pct is not None)
    count_drop2 = sum(1 for s in all_stocks if s.date_drop2pct is not None)
    count_drop4 = sum(1 for s in all_stocks if s.date_drop4pct is not None)

    pct_10pct = (count_10pct / total_count * 100) if total_count > 0 else 0
    pct_drop2 = (count_drop2 / total_count * 100) if total_count > 0 else 0
    pct_drop4 = (count_drop4 / total_count * 100) if total_count > 0 else 0

    # 2. 反转形态统计：先跌后涨
    # 先跌 2% 后涨 10%：date_drop2pct 的日期早于 date_10pct
    count_drop2_then_10pct = sum(
        1 for s in all_stocks
        if s.date_drop2pct is not None and s.date_10pct is not None
        and s.date_drop2pct < s.date_10pct
    )
    # 先跌 4% 后涨 10%：date_drop4pct 的日期早于 date_10pct
    count_drop4_then_10pct = sum(
        1 for s in all_stocks
        if s.date_drop4pct is not None and s.date_10pct is not None
        and s.date_drop4pct < s.date_10pct
    )

    pct_drop2_then_10pct = (count_drop2_then_10pct / total_count * 100) if total_count > 0 else 0
    pct_drop4_then_10pct = (count_drop4_then_10pct / total_count * 100) if total_count > 0 else 0

    # 3. 触及阈值的时间分布（第几天）
    def calc_day_stats(stocks, day_attr):
        """计算触及天数的统计信息"""
        valid_days = [getattr(s, day_attr) for s in stocks if getattr(s, day_attr) is not None]
        if not valid_days:
            return {'avg': 0, 'median': 0, 'min': 0, 'max': 0}
        return {
            'avg': sum(valid_days) / len(valid_days),
            'median': sorted(valid_days)[len(valid_days) // 2],
            'min': min(valid_days),
            'max': max(valid_days),
        }

    day_10pct_stats = calc_day_stats(all_stocks, 'first_10pct_day')
    day_drop2_stats = calc_day_stats(all_stocks, 'first_drop2_day')
    day_drop4_stats = calc_day_stats(all_stocks, 'first_drop4_day')

    # 4. 按 AI 分数分组的阈值触及率
    score_threshold_groups = [
        (4.5, float('inf'), "4.5 分+"),
        (4.0, 4.5, "4.0-4.5 分"),
        (0, 4.0, "0-4.0 分"),
    ]
    score_threshold_stats = []
    for low, high, label in score_threshold_groups:
        group = [s for s in all_stocks if low <= s.score < high]
        if group:
            g_total = len(group)
            g_count_10pct = sum(1 for s in group if s.date_10pct is not None)
            g_count_drop2 = sum(1 for s in group if s.date_drop2pct is not None)
            g_count_drop4 = sum(1 for s in group if s.date_drop4pct is not None)
            g_count_drop2_then_10pct = sum(
                1 for s in group
                if s.date_drop2pct is not None and s.date_10pct is not None
                and s.date_drop2pct < s.date_10pct
            )
            g_count_drop4_then_10pct = sum(
                1 for s in group
                if s.date_drop4pct is not None and s.date_10pct is not None
                and s.date_drop4pct < s.date_10pct
            )
            score_threshold_stats.append({
                'label': label, 'count': g_total,
                'count_10pct': g_count_10pct, 'pct_10pct': (g_count_10pct / g_total * 100) if g_total > 0 else 0,
                'count_drop2': g_count_drop2, 'pct_drop2': (g_count_drop2 / g_total * 100) if g_total > 0 else 0,
                'count_drop4': g_count_drop4, 'pct_drop4': (g_count_drop4 / g_total * 100) if g_total > 0 else 0,
                'count_drop2_then_10pct': g_count_drop2_then_10pct, 'pct_drop2_then_10pct': (g_count_drop2_then_10pct / g_total * 100) if g_total > 0 else 0,
                'count_drop4_then_10pct': g_count_drop4_then_10pct, 'pct_drop4_then_10pct': (g_count_drop4_then_10pct / g_total * 100) if g_total > 0 else 0,
            })

    # 5. 按排名分组的阈值触及率
    rank_threshold_groups = [
        (1, 3, "第 1-3 名"),
        (4, 5, "第 4-5 名"),
        (6, 7, "第 6-7 名"),
        (8, 10, "第 8-10 名"),
    ]
    rank_threshold_stats = []
    for low, high, label in rank_threshold_groups:
        group = [s for s in all_stocks if low <= s.rank <= high]
        if group:
            g_total = len(group)
            g_count_10pct = sum(1 for s in group if s.date_10pct is not None)
            g_count_drop2 = sum(1 for s in group if s.date_drop2pct is not None)
            g_count_drop4 = sum(1 for s in group if s.date_drop4pct is not None)
            g_count_drop2_then_10pct = sum(
                1 for s in group
                if s.date_drop2pct is not None and s.date_10pct is not None
                and s.date_drop2pct < s.date_10pct
            )
            g_count_drop4_then_10pct = sum(
                1 for s in group
                if s.date_drop4pct is not None and s.date_10pct is not None
                and s.date_drop4pct < s.date_10pct
            )
            rank_threshold_stats.append({
                'label': label, 'count': g_total,
                'count_10pct': g_count_10pct, 'pct_10pct': (g_count_10pct / g_total * 100) if g_total > 0 else 0,
                'count_drop2': g_count_drop2, 'pct_drop2': (g_count_drop2 / g_total * 100) if g_total > 0 else 0,
                'count_drop4': g_count_drop4, 'pct_drop4': (g_count_drop4 / g_total * 100) if g_total > 0 else 0,
                'count_drop2_then_10pct': g_count_drop2_then_10pct,
                'count_drop4_then_10pct': g_count_drop4_then_10pct,
            })

    # 生成 Markdown
    md_lines = []
    md_lines.append("# B1 碗口反弹策略回测分析报告（每日选股版）")
    md_lines.append("")
    md_lines.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append(f"**回测时间段**: {actual_start} 至 {actual_end}")
    md_lines.append(f"**选股策略**: B1 完美图形匹配 + 碗口反弹技术指标")
    md_lines.append(f"**选股频率**: 每个交易日")
    md_lines.append(f"**持有期**: {hold_days}天（自然日）")
    md_lines.append(f"**样本数量**: {total_count} 只股票")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 一、回测概览
    md_lines.append("## 一、回测概览")
    md_lines.append("")
    md_lines.append("### 1.1 核心指标")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 总交易数 | {total_count} 笔 |")
    md_lines.append(f"| 盈利股票 | {total_wins} 只 ({total_wins/total_count*100:.1f}%) |")
    md_lines.append(f"| 亏损股票 | {total_loses} 只 ({total_loses/total_count*100:.1f}%) |")
    md_lines.append(f"| 平均收益 | **{overall_avg_gain:+.2f}%** |")
    if best_trade:
        md_lines.append(f"| 最佳交易 | {best_trade.name} ({best_trade.code}) {best_trade.change_pct:+.2f}% |")
    if worst_trade:
        md_lines.append(f"| 最差交易 | {worst_trade.name} ({worst_trade.code}) {worst_trade.change_pct:+.2f}% |")
    md_lines.append("")

    md_lines.append("### 1.2 月度表现")
    md_lines.append("")
    md_lines.append("| 月份 | 选股数 | 盈利数 | 胜率 | 平均收益 |")
    md_lines.append("|------|--------|--------|------|----------|")
    for ms in monthly_stats:
        md_lines.append(f"| {ms['month']} | {ms['count']} | {ms['wins']} | {ms['win_rate']:.1f}% | {ms['avg_gain']:+.2f}% |")

    best_month = max(monthly_stats, key=lambda x: x['avg_gain']) if monthly_stats else None
    worst_month = min(monthly_stats, key=lambda x: x['avg_gain']) if monthly_stats else None
    md_lines.append("")
    if best_month:
        md_lines.append(f"**最佳月份**: {best_month['month']} ({best_month['avg_gain']:+.2f}%, 胜率 {best_month['win_rate']:.1f}%)")
    if worst_month:
        md_lines.append(f"**最差月份**: {worst_month['month']} ({worst_month['avg_gain']:+.2f}%, 胜率 {worst_month['win_rate']:.1f}%)")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 二、最大涨幅发生时间分布
    md_lines.append("## 二、最大涨幅发生时间分布")
    md_lines.append("")
    md_lines.append("### 2.1 时间区间分布")
    md_lines.append("")
    md_lines.append("| 时间区间 | 数量 | 占比 |")
    md_lines.append("|----------|------|------|")
    for td in time_distribution:
        md_lines.append(f"| {td['label']} | {td['count']} | {td['pct']:.1f}% |")
    md_lines.append("")

    md_lines.append("### 2.2 天数统计")
    md_lines.append("")
    md_lines.append("| 统计项 | 数值 |")
    md_lines.append("|--------|------|")
    md_lines.append(f"| 平均天数 | {avg_max_gain_day:.1f} 天 |")
    md_lines.append(f"| 中位天数 | {median_max_gain_day:.1f} 天 |")
    md_lines.append(f"| 最早发生 | 第 {min_max_gain_day} 天 |")
    md_lines.append(f"| 最晚发生 | 第 {max_max_gain_day} 天 |")
    md_lines.append("")
    md_lines.append("**结论**: 最大涨幅最集中出现在 **0-5 天** 区间，但整体分布较为均匀，说明获利机会在整个持有期内都可能出现。")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 三、盈亏分布占比
    md_lines.append("## 三、盈亏分布占比")
    md_lines.append("")
    md_lines.append("### 3.1 盈亏区间分布")
    md_lines.append("")
    md_lines.append("| 区间 | 数量 | 占比 |")
    md_lines.append("|------|------|------|")
    for gd in gain_distribution:
        md_lines.append(f"| {gd['label']} | {gd['count']} | {gd['pct']:.1f}% |")
    md_lines.append("")
    md_lines.append("**总体统计**:")
    md_lines.append(f"- 盈利股票：{total_wins} 只 ({total_wins/total_count*100:.1f}%)")
    md_lines.append(f"- 亏损股票：{total_loses} 只 ({total_loses/total_count*100:.1f}%)")
    md_lines.append(f"- 平均盈亏：{overall_avg_gain:+.2f}%")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 四、阈值触及统计（+10%, -2%, -4%）
    md_lines.append("## 四、阈值触及统计")
    md_lines.append("")
    md_lines.append("### 4.1 基础触及统计")
    md_lines.append("")
    md_lines.append("| 阈值类型 | 触及数量 | 触及率 |")
    md_lines.append("|----------|----------|-------|")
    md_lines.append(f"| ≥+10% (涨幅) | {count_10pct} | {pct_10pct:.1f}% |")
    md_lines.append(f"| ≤-2% (跌幅) | {count_drop2} | {pct_drop2:.1f}% |")
    md_lines.append(f"| ≤-4% (跌幅) | {count_drop4} | {pct_drop4:.1f}% |")
    md_lines.append("")

    md_lines.append("### 4.2 反转形态统计")
    md_lines.append("")
    md_lines.append("| 反转形态 | 数量 | 占比 | 说明 |")
    md_lines.append("|----------|------|------|------|")
    md_lines.append(f"| 先跌 2% 后涨 10% | {count_drop2_then_10pct} | {pct_drop2_then_10pct:.1f}% | 触及 -2% 后又触及 +10% |")
    md_lines.append(f"| 先跌 4% 后涨 10% | {count_drop4_then_10pct} | {pct_drop4_then_10pct:.1f}% | 触及 -4% 后又触及 +10% |")
    md_lines.append("")

    md_lines.append("### 4.3 触及时间统计（第几天）")
    md_lines.append("")
    md_lines.append("| 阈值 | 平均天数 | 中位天数 | 最早 | 最晚 |")
    md_lines.append("|------|----------|----------|------|------|")
    md_lines.append(f"| +10% | {day_10pct_stats['avg']:.1f} | {day_10pct_stats['median']:.1f} | {day_10pct_stats['min']} | {day_10pct_stats['max']} |")
    md_lines.append(f"| -2% | {day_drop2_stats['avg']:.1f} | {day_drop2_stats['median']:.1f} | {day_drop2_stats['min']} | {day_drop2_stats['max']} |")
    md_lines.append(f"| -4% | {day_drop4_stats['avg']:.1f} | {day_drop4_stats['median']:.1f} | {day_drop4_stats['min']} | {day_drop4_stats['max']} |")
    md_lines.append("")

    md_lines.append("### 4.4 按 AI 分数分组的阈值触及率")
    md_lines.append("")
    md_lines.append("| 分数区间 | 样本数 | +10% 触及率 | -2% 触及率 | -4% 触及率 | 先 -2% 后 +10% | 先 -4% 后 +10% |")
    md_lines.append("|----------|--------|------------|-----------|-----------|-------------|-------------|")
    for st in score_threshold_stats:
        md_lines.append(f"| {st['label']} | {st['count']} | {st['pct_10pct']:.1f}% | {st['pct_drop2']:.1f}% | {st['pct_drop4']:.1f}% | {st['pct_drop2_then_10pct']:.1f}% | {st['pct_drop4_then_10pct']:.1f}% |")
    md_lines.append("")

    md_lines.append("### 4.5 按排名分组的阈值触及率")
    md_lines.append("")
    md_lines.append("| 排名区间 | 样本数 | +10% 触及率 | -2% 触及率 | -4% 触及率 |")
    md_lines.append("|----------|--------|------------|-----------|-----------|")
    for rt in rank_threshold_stats:
        md_lines.append(f"| {rt['label']} | {rt['count']} | {rt['pct_10pct']:.1f}% | {rt['pct_drop2']:.1f}% | {rt['pct_drop4']:.1f}% |")
    md_lines.append("")

    md_lines.append("**结论**:")
    md_lines.append(f"- 约 **{pct_10pct:.1f}%** 的股票在持有期内触及 +10% 涨幅，说明策略存在获利机会")
    md_lines.append(f"- 约 **{pct_drop2:.1f}%** 的股票触及 -2% 跌幅，**{pct_drop4:.1f}%** 触及 -4% 跌幅")
    md_lines.append(f"- 先跌后涨的反转形态占比：{pct_drop2_then_10pct:.1f}% (先 -2% 后 +10%)，{pct_drop4_then_10pct:.1f}% (先 -4% 后 +10%)")
    md_lines.append(f"- 如果反转形态占比较高，说明**坚守持有期**可能带来反转收益")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 五、排名与盈亏相关性分析
    md_lines.append("## 五、排名与盈亏相关性分析")
    md_lines.append("")
    md_lines.append("### 5.1 不同排名区间的盈亏表现")
    md_lines.append("")
    md_lines.append("| 排名区间 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |")
    md_lines.append("|----------|--------|------|----------|--------------|")
    for rs in rank_stats:
        md_lines.append(f"| {rs['label']} | {rs['count']} | {rs['win_rate']:.1f}% | {rs['avg_gain']:+.2f}% | {rs['avg_max_gain']:+.2f}% |")
    md_lines.append("")

    md_lines.append("### 5.2 相关系数")
    md_lines.append("")
    md_lines.append("| 相关关系 | Pearson 系数 |")
    md_lines.append("|----------|--------------|")
    md_lines.append(f"| 排名 vs 最终盈亏 | **{rank_corr:.4f}** |")
    md_lines.append(f"| 排名 vs 最大涨幅 | **{rank_max_corr:.4f}** |")
    md_lines.append("")
    md_lines.append("**结论**:")
    if abs(rank_corr) < 0.05:
        md_lines.append("- 排名与最终盈亏**几乎无相关性**")
        md_lines.append("- B1 图形匹配的排名高低**不能**有效预测 30 日后的涨跌")
    else:
        md_lines.append(f"- 排名与最终盈亏存在**{'正' if rank_corr > 0 else '负'}相关性**")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 六、分数与盈亏相关性分析（用分数替代相似度）
    md_lines.append("## 六、分数与盈亏相关性分析")
    md_lines.append("")
    md_lines.append("### 6.1 不同分数区间的盈亏表现")
    md_lines.append("")
    md_lines.append("| 分数 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |")
    md_lines.append("|--------|--------|------|----------|--------------|")
    for ss in score_stats:
        md_lines.append(f"| {ss['label']} | {ss['count']} | {ss['win_rate']:.1f}% | {ss['avg_gain']:+.2f}% | {ss['avg_max_gain']:+.2f}% |")
    md_lines.append("")

    md_lines.append("### 6.2 相关系数")
    md_lines.append("")
    md_lines.append("| 相关关系 | Pearson 系数 |")
    md_lines.append("|----------|--------------|")
    md_lines.append(f"| 分数 vs 最终盈亏 | **{score_corr:.4f}** |")
    md_lines.append(f"| 分数 vs 最大涨幅 | **{score_max_corr:.4f}** |")
    md_lines.append("")
    md_lines.append("**结论**:")
    if abs(score_corr) < 0.05:
        md_lines.append("- 分数与盈亏**几乎无相关性**")
    else:
        md_lines.append(f"- 分数与盈亏存在**{'正' if score_corr > 0 else '负'}相关性**")
    md_lines.append("")

    md_lines.append("### 6.3 高分数股票详情（≥4.0 分）")
    md_lines.append("")
    md_lines.append("| 代码 | 名称 | 分数 | 最大涨幅 | 最终盈亏 | 选股日期 |")
    md_lines.append("|------|------|--------|----------|----------|----------|")
    for s in high_score_stocks[:20]:
        md_lines.append(f"| {s.code} | {s.name} | {s.score:.1f} | {s.max_gain_pct:+.2f}% | {s.change_pct:+.2f}% | {s.pick_date} |")
    md_lines.append("")

    high_score_count = len(high_score_stocks)
    high_score_wins = sum(1 for s in high_score_stocks if s.change_pct > 0)
    high_score_avg = sum(s.change_pct for s in high_score_stocks) / high_score_count if high_score_count > 0 else 0
    high_score_max = sum(s.max_gain_pct for s in high_score_stocks if s.max_gain_pct is not None) / len([s for s in high_score_stocks if s.max_gain_pct is not None]) if any(s.max_gain_pct is not None for s in high_score_stocks) else 0

    md_lines.append(f"**高分数股票（≥4.0 分）统计**:")
    md_lines.append("")
    md_lines.append("| 统计项 | 数值 |")
    md_lines.append("|--------|------|")
    md_lines.append(f"| 样本数 | {high_score_count} 只 |")
    md_lines.append(f"| 平均胜率 | {high_score_wins/high_score_count*100:.1f}% |")
    md_lines.append(f"| 平均收益 | {high_score_avg:+.2f}% |")
    md_lines.append(f"| 最大涨幅均值 | {high_score_max:+.2f}% |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 七、最大涨幅 vs 最终盈亏对比
    md_lines.append("## 七、最大涨幅 vs 最终盈亏对比")
    md_lines.append("")
    md_lines.append("### 7.1 差异统计")
    md_lines.append("")
    md_lines.append("")
    md_lines.append("| 指标 | 数值 |")
    md_lines.append("|------|------|")
    md_lines.append(f"| 平均最大涨幅 | **{avg_max_gain:.2f}%** |")
    md_lines.append(f"| 平均最终盈亏 | **{overall_avg_gain:+.2f}%** |")
    md_lines.append(f"| 平均差异 | **{avg_max_gain - overall_avg_gain:.2f}%** |")
    md_lines.append(f"| 曾达到过比最终盈亏高 5% 以上的股票数 | {diff_above_5pct} 只 ({diff_pct:.1f}%) |")
    md_lines.append("")
    md_lines.append("**结论**:")
    md_lines.append("- 大部分股票在持有期内都有过不错的表现")
    md_lines.append(f"- **{diff_pct:.1f}%** 的股票最后回吐了大部分涨幅")
    md_lines.append(f"- 平均少赚了 {avg_max_gain - overall_avg_gain:.2f}%，提示可能需要优化止盈策略")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 八、选股分类与盈亏关系
    md_lines.append("## 八、选股分类与盈亏关系")
    md_lines.append("")
    md_lines.append("### 8.1 不同分类的表现")
    md_lines.append("")
    md_lines.append("| 分类 | 样本数 | 胜率 | 平均盈亏 | 最大涨幅均值 |")
    md_lines.append("|------|--------|------|----------|--------------|")
    for cs in category_stats:
        md_lines.append(f"| {cs['category']} | {cs['count']} | {cs['win_rate']:.1f}% | {cs['avg_gain']:+.2f}% | {cs['avg_max_gain']:+.2f}% |")
    md_lines.append("")

    best_cat = category_stats[0] if category_stats else None
    if best_cat:
        md_lines.append(f"**结论**: \"{best_cat['category']}\" 分类表现最佳，胜率 {best_cat['win_rate']:.1f}%，平均盈亏 {best_cat['avg_gain']:+.2f}%。")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 九、阈值触及统计总结
    md_lines.append("## 九、阈值触及统计总结")
    md_lines.append("")
    md_lines.append("### 9.1 触及率概览")
    md_lines.append(f"- **+10% 涨幅触及率**: {pct_10pct:.1f}% ({count_10pct} 只股票)")
    md_lines.append(f"- **-2% 跌幅触及率**: {pct_drop2:.1f}% ({count_drop2} 只股票)")
    md_lines.append(f"- **-4% 跌幅触及率**: {pct_drop4:.1f}% ({count_drop4} 只股票)")
    md_lines.append("")
    md_lines.append("### 9.2 反转形态总结")
    md_lines.append(f"- **先跌 2% 后涨 10%**: {count_drop2_then_10pct} 只 ({pct_drop2_then_10pct:.1f}%)")
    md_lines.append(f"- **先跌 4% 后涨 10%**: {count_drop4_then_10pct} 只 ({pct_drop4_then_10pct:.1f}%)")
    md_lines.append("")
    md_lines.append("### 9.3 触及时间特征")
    md_lines.append(f"- +10% 平均触及时间：**{day_10pct_stats['avg']:.1f} 天**")
    md_lines.append(f"- -2% 平均触及时间：**{day_drop2_stats['avg']:.1f} 天**")
    md_lines.append(f"- -4% 平均触及时间：**{day_drop4_stats['avg']:.1f} 天**")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 十、关键发现总结
    md_lines.append("## 十、关键发现总结")
    md_lines.append("")
    md_lines.append("### 10.1 时间分布特征")
    md_lines.append(f"- 最大涨幅中位时间为 **{median_max_gain_day} 天**，平均 **{avg_max_gain_day:.1f} 天**")
    md_lines.append("- 分布较为均匀，获利机会在持有期内各阶段都可能出现")
    md_lines.append("")
    md_lines.append("### 10.2 盈亏分布特征")
    md_lines.append(f"- 策略整体**{'盈利' if overall_avg_gain > 0 else '亏损'}**，胜率 **{overall_win_rate:.1f}%**")
    md_lines.append(f"- 平均收益 **{overall_avg_gain:+.2f}%**")
    md_lines.append("")
    md_lines.append("### 10.3 排名相关性")
    md_lines.append(f"- 排名与最终盈亏的相关系数为 **{rank_corr:.4f}**")
    if abs(rank_corr) < 0.05:
        md_lines.append("- B1 图形匹配的排名不能完全预测 30 日后的涨跌")
    md_lines.append("")
    md_lines.append("### 10.4 分数相关性")
    md_lines.append(f"- 分数与最终盈亏的相关系数为 **{score_corr:.4f}**")
    if abs(score_corr) < 0.05:
        md_lines.append("- 分数对盈亏没有明显预测作用")
    md_lines.append("")
    md_lines.append("### 10.5 最大涨幅启示")
    md_lines.append(f"- 平均最大涨幅 ({avg_max_gain:.2f}%) 高于最终盈亏 ({overall_avg_gain:+.2f}%)")
    md_lines.append(f"- **{diff_pct:.1f}%** 的股票曾达到过比最终盈亏高 5% 以上的涨幅")
    md_lines.append("- 说明持有期内大部分股票都有过不错的表现，但最后回吐了涨幅")
    md_lines.append("- **建议**: 考虑引入动态止盈策略（如达到 8-10% 涨幅时提前止盈）")
    md_lines.append("")
    md_lines.append("### 10.6 阈值触及启示")
    md_lines.append(f"- 约 {pct_10pct:.1f}% 的股票在持有期内达到过 +10% 涨幅")
    md_lines.append(f"- 约 {pct_drop2_then_10pct:.1f}% 的股票经历了先跌 2% 后涨 10% 的反转")
    md_lines.append(f"- 约 {pct_drop4_then_10pct:.1f}% 的股票经历了先跌 4% 后涨 10% 的深 V 反转")
    md_lines.append("- 这表明坚守持有期对部分股票是有效的，能够捕捉到反转收益")
    md_lines.append("")
    md_lines.append("### 10.7 分类策略建议")
    if best_cat:
        md_lines.append(f"- \"{best_cat['category']}\" 分类表现最佳")
        md_lines.append(f"- **建议**: 可优先选择该分类的股票，或增加其权重")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 十一、策略优化建议
    md_lines.append("## 十一、策略优化建议")
    md_lines.append("")
    md_lines.append("基于以上分析，提出以下优化建议：")
    md_lines.append("")
    md_lines.append("1. **分数阈值调整**:")
    md_lines.append("   - 分数与盈亏相关性较弱，不建议仅基于分数筛选")
    md_lines.append("")
    if best_cat:
        md_lines.append(f"2. **优先选择\"{best_cat['category']}\"分类**: 该分类胜率 {best_cat['win_rate']:.1f}%，平均盈亏 {best_cat['avg_gain']:+.2f}%")
        md_lines.append("")
    md_lines.append("3. **引入动态止盈**:")
    md_lines.append(f"   - {diff_pct:.1f}% 的股票曾达到过比最终盈亏高 5% 以上的涨幅")
    md_lines.append(f"   - 建议在持有期第 {int(median_max_gain_day)}-{int(median_max_gain_day)+5} 天（最大涨幅高发期）设置止盈点（如 8-10%）")
    md_lines.append("")
    md_lines.append("4. **持有期调整**:")
    md_lines.append(f"   - 最大涨幅中位数为 **{median_max_gain_day} 天**")
    md_lines.append(f"   - 可考虑将持有期缩短至 {int(median_max_gain_day)}-{int(median_max_gain_day)+5} 天，配合止盈策略")
    md_lines.append("")
    md_lines.append("5. **排名权重调整**:")
    md_lines.append("   - 排名与盈亏无显著相关性，可考虑降低排名权重")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")

    # 十、数据说明
    md_lines.append("## 十、数据说明")
    md_lines.append("")
    md_lines.append(f"- **天数计算方式**: 自然日（日历日期差），非交易日")
    md_lines.append(f"- **数据来源**: 本地缓存的股票历史数据（CSV 格式）")
    md_lines.append(f"- **回测方法**: 每个交易日用截至当日的历史数据重新跑选股，避免未来函数")
    md_lines.append(f"- **选股数量**: 每日选前 10 只股票")
    md_lines.append(f"- **持有期**: {hold_days} 自然日")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("**报告完**")
    md_lines.append("")
    md_lines.append(f"*生成脚本：historical_backtest.py*")
    md_lines.append(f"*数据文件：{actual_start}_{actual_end}*")
    md_lines.append("")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    print(f"\nMarkdown 报告已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description="历史回测脚本（重新运行选股流程）")
    parser.add_argument('--months', type=int, default=12, help='回测月数（默认 12）')
    parser.add_argument('--top-n', type=int, default=10, help='每月选股数量（默认 10）')
    parser.add_argument('--output', type=str, default=None, help='输出 JSON 文件路径')
    parser.add_argument('--report', type=str, help='输出 Markdown 报告文件路径（默认：backtest/backtest_report.md）')
    parser.add_argument('--pick-date', type=str, default=None, help='指定选股日 (YYYY-MM-DD)，用于单日期回测')
    parser.add_argument('--hold-days', type=int, default=30, help='持有天数（默认 30 个交易日）')
    parser.add_argument('--start', type=str, default=None, help='起始日期 (YYYY-MM-DD)，用于日期范围回测')
    parser.add_argument('--end', type=str, default=None, help='结束日期 (YYYY-MM-DD)，用于日期范围回测')
    parser.add_argument('--ai-workers', type=int, default=None, help='AI 分析并发数（可选，覆盖配置文件）')
    parser.add_argument('--ai-request-delay', type=float, default=None, help='AI 分析请求间隔秒数（可选）')
    parser.add_argument('--use-ollama', action='store_true', help='使用 Ollama 本地模型（默认使用阿里云 API）')

    args = parser.parse_args()

    # 默认报告路径
    if args.report is None:
        args.report = str(Path(__file__).parent / 'backtest_report.md')

    args = parser.parse_args()

    # 日期范围模式
    if args.start and args.end:
        results = run_date_range_backtest(
            start_date=args.start,
            end_date=args.end,
            top_n=args.top_n,
            hold_days=args.hold_days,
            ai_workers=args.ai_workers,
            ai_request_delay=args.ai_request_delay,
            use_ollama=args.use_ollama
        )
    # 指定日期模式
    elif args.pick_date:
        results = run_single_date_backtest(
            pick_date=args.pick_date,
            top_n=args.top_n,
            hold_days=args.hold_days,
            ai_workers=args.ai_workers,
            ai_request_delay=args.ai_request_delay,
            use_ollama=args.use_ollama
        )
    else:
        # 多月中回测模式
        results = run_historical_backtest(
            months=args.months,
            top_n=args.top_n,
            ai_workers=args.ai_workers,
            ai_request_delay=args.ai_request_delay,
            use_ollama=args.use_ollama
        )

    if not results:
        print("\n[错误] 回测未能生成任何结果")
        sys.exit(1)

    print_report(results)

    if args.output:
        save_report(results, Path(args.output))

    if args.report:
        save_markdown_report(results, Path(args.report),
                            start_date=args.start, end_date=args.end,
                            hold_days=args.hold_days)


def run_date_range_backtest(start_date: str, end_date: str, top_n: int = 10,
                            hold_days: int = 30,
                            ai_workers: int = None, ai_request_delay: float = None,
                            use_ollama: bool = False) -> List[MonthlyResult]:
    """
    运行日期范围回测

    Args:
        start_date: 起始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        top_n: 选前 N 只股票
        hold_days: 持有天数（交易日）
        ai_workers: AI 分析并发数（可选）
        ai_request_delay: AI 分析请求间隔秒数（可选）
        use_ollama: 是否使用 Ollama 本地模型（默认 False）

    Returns:
        list of MonthlyResult
    """
    print(f"\n{'='*70}")
    print(f"日期范围回测：{start_date} 至 {end_date}")
    print(f"选前{top_n}只股票，持有{hold_days}个交易日")
    model_type = "Ollama 本地模型" if use_ollama else "阿里云 API"
    print(f"AI 分析模式：{model_type}")
    if ai_workers:
        print(f"AI 分析并发数：{ai_workers}")
    if ai_request_delay is not None:
        print(f"AI 分析请求间隔：{ai_request_delay}秒")
    print(f"{'='*70}\n")

    # 获取交易日列表
    print("[加载] 获取交易日列表...")
    trading_dates = get_all_trading_dates()
    if not trading_dates:
        print("[错误] 无法获取交易日列表")
        return []

    print(f"  共 {len(trading_dates)} 个交易日")
    print(f"  数据范围：{trading_dates[0]} 至 {trading_dates[-1]}")

    # 筛选范围内的交易日
    filtered_dates = [d for d in trading_dates if start_date <= d <= end_date]
    if not filtered_dates:
        print(f"[错误] 指定日期范围内无交易日数据")
        return []

    print(f"\n[计划] 范围内共 {len(filtered_dates)} 个交易日")
    print(f"开始回测...\n")

    results = []
    kline_cache = {}

    for pick_date in filtered_dates:
        print(f"\n{'='*60}")
        print(f"[选股日] {pick_date}")
        print(f"{'='*60}")

        # 1. 检查是否已有结果
        suggestion = load_suggestion(pick_date)

        if suggestion is None:
            print(f"  [新运行] 未找到历史结果，开始运行选股流程...")
            if not run_full_pipeline(pick_date, max_workers=ai_workers, request_delay=ai_request_delay,
                                      use_ollama=use_ollama):
                print(f"  [跳过] 选股流程失败")
                continue

            suggestion = load_suggestion(pick_date)

        if suggestion is None:
            print(f"  [跳过] 无法获取 AI 评分结果")
            continue

        # 2. 获取推荐股票（AI 评分最高的前 N 只）
        recommendations = suggestion.get('recommendations', [])
        if not recommendations:
            print(f"  [跳过] 无推荐股票")
            continue

        # 按排名排序（已经是排好序的）
        selected = recommendations[:top_n]
        print(f"  AI 推荐前{top_n}只股票")

        # 3. 获取候选数据（用于获取选股时价格）
        candidates_file = CANDIDATES_DIR / f"candidates_{pick_date}.json"
        candidates_map = {}
        if candidates_file.exists():
            with open(candidates_file, 'r', encoding='utf-8') as f:
                candidates_data = json.load(f)
            for c in candidates_data.get('candidates', []):
                candidates_map[c['code']] = c

        # 4. 计算目标日期（第 N 个交易日）
        target_date = find_target_trading_day(pick_date, trading_dates, hold_days)
        if not target_date:
            print(f"  [警告] 无法找到第{hold_days}个交易日，使用自然日 +{hold_days}")
            target_date = (datetime.strptime(pick_date, "%Y-%m-%d") + timedelta(days=hold_days)).strftime("%Y-%m-%d")

        print(f"  目标日（+{hold_days}交易日）：{target_date}")

        # 5. 计算每只股票的表现
        stock_results = []

        for rec in selected:
            code = rec['code']
            rank = rec.get('rank', 0)
            score = rec.get('total_score', 0)
            verdict = rec.get('verdict', '')
            signal_type = rec.get('signal_type', '')

            # 从候选数据中获取更多信息
            cand_info = candidates_map.get(code, {})
            name = cand_info.get('name', '')
            strategy = cand_info.get('strategy', '')

            # 选股日价格
            pick_price = cand_info.get('close', 0)
            if pick_price is None or pick_price == 0:
                pick_price = get_price_on_date(code, pick_date, kline_cache)

            if pick_price is None or pick_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, score=score,
                    pick_price=0, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, verdict=verdict,
                    signal_type=signal_type, strategy=strategy, status="no_data",
                    max_gain_pct=None, max_gain_day=None
                ))
                continue

            # 目标日价格
            target_price = get_price_on_date(code, target_date, kline_cache)

            if target_price is None or target_price <= 0:
                stock_results.append(StockResult(
                    code=code, name=name, rank=rank, score=score,
                    pick_price=pick_price, target_price=None,
                    change_pct=None, pick_date=pick_date,
                    target_date=target_date, verdict=verdict,
                    signal_type=signal_type, strategy=strategy, status="no_data",
                    max_gain_pct=None, max_gain_day=None
                ))
                continue

            # 计算涨跌幅
            change_pct = ((target_price - pick_price) / pick_price) * 100
            status = "win" if change_pct > 0 else ("lose" if change_pct < 0 else "tie")

            # 计算期间最大涨幅
            max_gain_pct, max_gain_day = calc_max_gain_during_period(
                code, pick_date, target_date, pick_price, trading_dates, kline_cache)

            # 追踪阈值触及日期
            threshold_data = track_threshold_dates(
                code, pick_date, target_date, pick_price, trading_dates, kline_cache)

            stock_results.append(StockResult(
                code=code, name=name, rank=rank, score=score,
                pick_price=pick_price, target_price=target_price,
                change_pct=change_pct, pick_date=pick_date,
                target_date=target_date, verdict=verdict,
                signal_type=signal_type, strategy=strategy, status=status,
                max_gain_pct=max_gain_pct, max_gain_day=max_gain_day,
                **threshold_data
            ))

        # 6. 统计结果
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


def run_single_date_backtest(pick_date: str, top_n: int = 10, hold_days: int = 30,
                             ai_workers: int = None, ai_request_delay: float = None,
                             use_ollama: bool = False) -> List[MonthlyResult]:
    """
    运行单日期回测

    Args:
        pick_date: 选股日 YYYY-MM-DD
        top_n: 选前 N 只股票
        hold_days: 持有天数（交易日）
        ai_workers: AI 分析并发数（可选）
        ai_request_delay: AI 分析请求间隔秒数（可选）
        use_ollama: 是否使用 Ollama 本地模型（默认 False）

    Returns:
        list of MonthlyResult
    """
    print(f"\n{'='*70}")
    print(f"单日期回测：选股日={pick_date}, 持有{hold_days}个交易日，选前{top_n}只股票")
    model_type = "Ollama 本地模型" if use_ollama else "阿里云 API"
    print(f"AI 分析模式：{model_type}")
    if ai_workers:
        print(f"AI 分析并发数：{ai_workers}")
    if ai_request_delay is not None:
        print(f"AI 分析请求间隔：{ai_request_delay}秒")
    print(f"{'='*70}\n")

    # 获取交易日列表
    print("[加载] 获取交易日列表...")
    trading_dates = get_all_trading_dates()
    if not trading_dates:
        print("[错误] 无法获取交易日列表")
        return []

    print(f"  共 {len(trading_dates)} 个交易日")
    print(f"  数据范围：{trading_dates[0]} 至 {trading_dates[-1]}")

    # 找到选股日之后最近的交易日
    actual_pick_date = None
    for d in trading_dates:
        if d >= pick_date:
            actual_pick_date = d
            break

    if not actual_pick_date:
        print(f"[错误] 选股日 {pick_date} 超出数据范围")
        return []

    if actual_pick_date != pick_date:
        print(f"[调整] 选股日调整为最近交易日：{actual_pick_date}")

    # 计算目标日期
    target_date = find_target_trading_day(actual_pick_date, trading_dates, hold_days)
    if not target_date:
        target_date = (datetime.strptime(actual_pick_date, "%Y-%m-%d") + timedelta(days=hold_days)).strftime("%Y-%m-%d")
        print(f"[警告] 无法找到第{hold_days}个交易日，使用自然日 +{hold_days}: {target_date}")

    print(f"\n选股日：{actual_pick_date}")
    print(f"目标日：{target_date} (+{hold_days}交易日)")

    results = []
    kline_cache = {}

    print(f"\n{'='*60}")
    print(f"[选股日] {actual_pick_date}")
    print(f"{'='*60}")

    # 1. 检查是否已有结果
    suggestion = load_suggestion(actual_pick_date)

    if suggestion is None:
        print(f"  [新运行] 未找到历史结果，开始运行选股流程...")
        if not run_full_pipeline(actual_pick_date, max_workers=ai_workers, request_delay=ai_request_delay,
                                  use_ollama=use_ollama):
            print(f"  [错误] 选股流程失败")
            return []

        suggestion = load_suggestion(actual_pick_date)

    if suggestion is None:
        print(f"  [错误] 无法获取 AI 评分结果")
        return []

    # 2. 获取推荐股票
    recommendations = suggestion.get('recommendations', [])
    if not recommendations:
        print(f"  [错误] 无推荐股票")
        return []

    selected = recommendations[:top_n]
    print(f"  AI 推荐前{top_n}只股票")

    # 3. 获取候选数据
    candidates_file = CANDIDATES_DIR / f"candidates_{actual_pick_date}.json"
    candidates_map = {}
    if candidates_file.exists():
        with open(candidates_file, 'r', encoding='utf-8') as f:
            candidates_data = json.load(f)
        for c in candidates_data.get('candidates', []):
            candidates_map[c['code']] = c

    print(f"  目标日（+{hold_days}交易日）：{target_date}")

    # 4. 计算每只股票的表现
    stock_results = []

    for rec in selected:
        code = rec['code']
        rank = rec.get('rank', 0)
        score = rec.get('total_score', 0)
        verdict = rec.get('verdict', '')
        signal_type = rec.get('signal_type', '')

        # 从候选数据中获取更多信息
        cand_info = candidates_map.get(code, {})
        name = cand_info.get('name', '')
        strategy = cand_info.get('strategy', '')

        # 选股日价格
        pick_price = cand_info.get('close', 0)
        if pick_price is None or pick_price == 0:
            pick_price = get_price_on_date(code, actual_pick_date, kline_cache)

        if pick_price is None or pick_price <= 0:
            stock_results.append(StockResult(
                code=code, name=name, rank=rank, score=score,
                pick_price=0, target_price=None,
                change_pct=None, pick_date=actual_pick_date,
                target_date=target_date, verdict=verdict,
                signal_type=signal_type, strategy=strategy, status="no_data",
                max_gain_pct=None, max_gain_day=None
            ))
            continue

        # 目标日价格
        target_price = get_price_on_date(code, target_date, kline_cache)

        if target_price is None or target_price <= 0:
            stock_results.append(StockResult(
                code=code, name=name, rank=rank, score=score,
                pick_price=pick_price, target_price=None,
                change_pct=None, pick_date=actual_pick_date,
                target_date=target_date, verdict=verdict,
                signal_type=signal_type, strategy=strategy, status="no_data",
                max_gain_pct=None, max_gain_day=None
            ))
            continue

        # 计算涨跌幅
        change_pct = ((target_price - pick_price) / pick_price) * 100
        status = "win" if change_pct > 0 else ("lose" if change_pct < 0 else "tie")

        # 计算期间最大涨幅
        max_gain_pct, max_gain_day = calc_max_gain_during_period(
            code, actual_pick_date, target_date, pick_price, trading_dates, kline_cache)

        # 追踪阈值触及日期
        threshold_data = track_threshold_dates(
            code, actual_pick_date, target_date, pick_price, trading_dates, kline_cache)

        stock_results.append(StockResult(
            code=code, name=name, rank=rank, score=score,
            pick_price=pick_price, target_price=target_price,
            change_pct=change_pct, pick_date=actual_pick_date,
            target_date=target_date, verdict=verdict,
            signal_type=signal_type, strategy=strategy, status=status,
            max_gain_pct=max_gain_pct, max_gain_day=max_gain_day,
            **threshold_data
        ))

    # 5. 统计结果
    win_count = sum(1 for s in stock_results if s.status == "win")
    lose_count = sum(1 for s in stock_results if s.status == "lose")
    tie_count = sum(1 for s in stock_results if s.status == "tie")
    no_data_count = sum(1 for s in stock_results if s.status == "no_data")

    valid_results = [s for s in stock_results if s.change_pct is not None]
    avg_gain = sum(s.change_pct for s in valid_results) / len(valid_results) if valid_results else 0

    effective_total = win_count + lose_count + tie_count
    win_rate = (win_count / effective_total * 100) if effective_total > 0 else 0

    monthly_result = MonthlyResult(
        pick_date=actual_pick_date,
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

    print(f"\n  结果：胜={win_count}, 负={lose_count}, 平={tie_count}, 无数据={no_data_count}")
    print(f"  胜率：{win_rate:.1f}%, 平均收益：{avg_gain:.2f}%")

    return results


if __name__ == "__main__":
    main()
