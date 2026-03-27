"""
backtest/data_loader.py
~~~~~~~~~~~~~~~~~~~~~~~
回测数据加载工具

提供统一的 K 线数据读取接口，自动支持 CSV 和 Parquet 格式

注意：以下函数保持在各自脚本中本地定义：
- get_monthly_dates: 在 monthly_backtest.py 中
- find_first_trading_day_of_month: 在 historical_backtest.py 中
- find_target_trading_day: 在 historical_backtest.py 中
- calc_max_gain_during_period: 在 historical_backtest.py 中
- track_threshold_dates: 在 historical_backtest.py 中
"""
from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def get_all_trading_dates(code: str = "000001") -> List[str]:
    """
    获取所有交易日列表（从 K 线数据中提取）

    使用 CSVManager 统一读取，自动支持 CSV/Parquet 格式
    """
    from pipeline.csv_manager import CSVManager

    csv_manager = CSVManager(str(RAW_DIR))
    df = csv_manager.read_stock(code)

    if df is None or df.empty:
        # 降级：尝试直接读取 CSV
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

    # 从 DataFrame 提取日期
    dates = df['date'].astype(str).tolist()
    return sorted(dates)


def find_first_trading_day(year: int, month: int, trading_dates: List[str]) -> Optional[str]:
    """找到指定月份第一个交易日"""
    target = f"{year:04d}-{month:02d}-01"
    for d in trading_dates:
        if d >= target and d.startswith(f"{year:04d}-{month:02d}"):
            return d
    return None


def find_target_day(pick_date: str, trading_dates: List[str], days: int = 30) -> Optional[str]:
    """
    找到第 N 个交易日

    Args:
        pick_date: 选股日期
        trading_dates: 交易日列表
        days: 第 N 个交易日（默认 30）

    Returns:
        目标交易日日期
    """
    for i, d in enumerate(trading_dates):
        if d >= pick_date:
            if i + days < len(trading_dates):
                return trading_dates[i + days]
            return None
    return None


def load_kline_data(code: str) -> Dict[str, dict]:
    """
    加载单只股票的 K 线数据

    使用 CSVManager 统一读取，自动支持 CSV/Parquet 格式

    Returns:
        {date: {open, close, high, low, volume}}
    """
    from pipeline.csv_manager import CSVManager

    csv_manager = CSVManager(str(RAW_DIR))
    df = csv_manager.read_stock(code)

    if df is None or df.empty:
        # 降级：尝试直接读取 CSV
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

    # 从 DataFrame 构建字典
    data = {}
    for _, row in df.iterrows():
        date_str = str(row['date'])
        if hasattr(row['date'], 'strftime'):
            date_str = row['date'].strftime('%Y-%m-%d')

        data[date_str] = {
            'open': float(row.get('open', 0)),
            'close': float(row.get('close', 0)),
            'high': float(row.get('high', 0)),
            'low': float(row.get('low', 0)),
            'volume': float(row.get('volume', 0))
        }

    return data


def get_price_on_date(code: str, target_date: str, kline_cache: dict) -> Optional[float]:
    """
    获取股票在指定日期的收盘价

    如果目标日期没有数据，找最近的交易日（向前最多 10 天）

    Args:
        code: 股票代码
        target_date: 目标日期 YYYY-MM-DD
        kline_cache: K 线数据缓存

    Returns:
        收盘价，无法获取返回 None
    """
    # 先用缓存
    if code in kline_cache:
        kline_data = kline_cache[code]
    else:
        kline_data = load_kline_data(code)
        kline_cache[code] = kline_data

    if not kline_data:
        return None

    # 精确匹配
    if target_date in kline_data:
        return kline_data[target_date]['close']

    # 找最近的交易日（向前找最多 10 天）
    target_dt = datetime.strptime(target_date, "%Y-%m-%d")
    for i in range(10):
        check_date = (target_dt - timedelta(days=i)).strftime("%Y-%m-%d")
        if check_date in kline_data:
            return kline_data[check_date]['close']

    return None


def get_price_range(code: str, start_date: str, end_date: str,
                    kline_cache: dict) -> List[tuple]:
    """
    获取指定日期范围内的 K 线数据

    Args:
        code: 股票代码
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        kline_cache: K 线数据缓存

    Returns:
        [(date, open, close, high, low, volume), ...] 按日期正序排列
    """
    if code not in kline_cache:
        kline_cache[code] = load_kline_data(code)

    kline_data = kline_cache[code]
    if not kline_data:
        return []

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    results = []
    current_dt = start_dt

    while current_dt <= end_dt:
        date_str = current_dt.strftime("%Y-%m-%d")
        if date_str in kline_data:
            data = kline_data[date_str]
            results.append((
                date_str,
                data['open'],
                data['close'],
                data['high'],
                data['low'],
                data['volume']
            ))
        current_dt += timedelta(days=1)

    return results


def calc_max_gain(code: str, pick_date: str, target_date: str,
                  pick_price: float, trading_dates: List[str],
                  kline_cache: dict) -> tuple:
    """
    计算期间最大涨幅

    Returns:
        (max_gain_pct, max_gain_day) 最大涨幅百分比和发生日期索引
    """
    if code not in kline_cache:
        kline_cache[code] = load_kline_data(code)

    kline_data = kline_cache[code]
    if not kline_data:
        return None, None

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

    max_price = pick_price
    max_day_idx = 0

    for i in range(start_idx, min(end_idx + 1, len(trading_dates))):
        d = trading_dates[i]
        if d in kline_data:
            price = kline_data[d]['close']
            if price > max_price:
                max_price = price
                max_day_idx = i - start_idx

    if max_price <= pick_price:
        return 0.0, 0

    max_gain_pct = ((max_price - pick_price) / pick_price) * 100
    return max_gain_pct, max_day_idx


def calc_threshold_dates(code: str, pick_date: str, target_date: str,
                         pick_price: float, trading_dates: List[str],
                         kline_cache: dict) -> dict:
    """
    计算阈值触及日期

    Returns:
        {
            'date_10pct': 第一次达到 +10% 的日期,
            'date_drop2pct': 第一次达到 -2% 的日期,
            'date_drop4pct': 第一次达到 -4% 的日期,
            'first_10pct_day': +10% 发生在第几天,
            'first_drop2_day': -2% 发生在第几天,
            'first_drop4_day': -4% 发生在第几天,
        }
    """
    result = {
        'date_10pct': None,
        'date_drop2pct': None,
        'date_drop4pct': None,
        'first_10pct_day': None,
        'first_drop2_day': None,
        'first_drop4_day': None,
    }

    if code not in kline_cache:
        kline_cache[code] = load_kline_data(code)

    kline_data = kline_cache[code]
    if not kline_data:
        return result

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

    threshold_10pct = pick_price * 1.10
    threshold_drop2 = pick_price * 0.98
    threshold_drop4 = pick_price * 0.96

    for i in range(start_idx, min(end_idx + 1, len(trading_dates))):
        d = trading_dates[i]
        if d in kline_data:
            price = kline_data[d]['close']
            day_num = i - start_idx

            # 检查 +10%
            if result['date_10pct'] is None and price >= threshold_10pct:
                result['date_10pct'] = d
                result['first_10pct_day'] = day_num

            # 检查 -2%
            if result['date_drop2pct'] is None and price <= threshold_drop2:
                result['date_drop2pct'] = d
                result['first_drop2_day'] = day_num

            # 检查 -4%
            if result['date_drop4pct'] is None and price <= threshold_drop4:
                result['date_drop4pct'] = d
                result['first_drop4_day'] = day_num

    return result
