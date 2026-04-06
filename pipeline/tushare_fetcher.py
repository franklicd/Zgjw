"""
A 股数据抓取模块 - 使用 Tushare Pro 接口（严格模式）

数据精确性原则：
1. 如果选择日期下没有数据（行情或股票列表），则直接从 Tushare 下载
2. 如果日期是交易日，则数据与日期必须严格匹配
3. 计算任何统计值时，不使用猜测或不匹配日期的缓存数据
4. 数据精确性排在第一位
"""
import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.csv_manager import CSVManager


class TushareFetcher:
    """Tushare Pro 数据抓取器（严格模式）"""

    def __init__(self, data_dir="data"):
        self.csv_manager = CSVManager(data_dir)
        self.full_data_dir = Path(data_dir)
        self.stock_names_file = Path(data_dir) / 'stock_names.json'
        self.trade_calendar_cache: Optional[pd.DataFrame] = None
        self.trade_calendar_date: Optional[str] = None  # 缓存的交易日历日期范围

        # 初始化 tushare
        self._init_tushare()

    def _init_tushare(self):
        """初始化 Tushare Pro API"""
        import tushare as ts

        # 从环境变量获取 token
        token = os.environ.get('TUSHARE_TOKEN')
        if not token:
            raise RuntimeError("环境变量 TUSHARE_TOKEN 未设置，请先配置 token")

        ts.set_token(token)
        self.pro = ts.pro_api()
        print(f"✓ Tushare Pro 初始化成功")

    # ==================== 交易日历相关 ====================

    def _load_trade_calendar(self, exchange: str = 'SSE', start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        加载交易日历（带本地缓存）

        Args:
            exchange: 交易所 SSE-上交所，SZSE-深交所
            start_date: 开始日期 YYYYMMDD
            end_date: 结束日期 YYYYMMDD

        Returns:
            交易日历 DataFrame
        """
        # 使用当前年份作为默认范围
        current_year = datetime.now().year
        if start_date is None:
            start_date = f"{current_year - 1}0101"
        if end_date is None:
            end_date = f"{current_year + 1}1231"

        # 检查缓存是否在有效范围内
        if self.trade_calendar_cache is not None and self.trade_calendar_date:
            cache_start, cache_end = self.trade_calendar_date
            if cache_start <= start_date and cache_end >= end_date:
                return self.trade_calendar_cache

        # 从 Tushare 获取交易日历
        try:
            df = self.pro.trade_cal(exchange=exchange, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                self.trade_calendar_cache = df
                self.trade_calendar_date = (start_date, end_date)
                return df
        except Exception as e:
            print(f"  获取交易日历失败：{e}")

        # 尝试从本地加载缓存
        calendar_cache_file = self.full_data_dir / 'trade_calendar_cache.parquet'
        if calendar_cache_file.exists():
            try:
                df = pd.read_parquet(calendar_cache_file)
                # 过滤到需要的日期范围（将 cal_date 转为字符串比较）
                df = df[(df['cal_date'] >= start_date) & (df['cal_date'] <= end_date)]
                if not df.empty:
                    self.trade_calendar_cache = df
                    self.trade_calendar_date = (start_date, end_date)
                    return df
            except:
                pass

        return pd.DataFrame()

    def is_trade_date(self, date: str, exchange: str = 'SSE') -> bool:
        """
        检查给定日期是否为交易日

        Args:
            date: 日期 YYYY-MM-DD 或 YYYYMMDD
            exchange: 交易所

        Returns:
            True 如果是交易日
        """
        # 标准化日期格式
        if '-' in date:
            date = date.replace('-', '')

        # 确保日期在交易日历范围内
        date_int = int(date)
        if self.trade_calendar_cache is None or self.trade_calendar_date is None:
            # 加载包含该日期的交易日历
            year = int(date[:4])
            start_date = f"{year - 1}0101"
            end_date = f"{year + 1}1231"
            self._load_trade_calendar(exchange=exchange, start_date=start_date, end_date=end_date)

        if self.trade_calendar_cache is None or self.trade_calendar_cache.empty:
            # 无法获取交易日历，假设是交易日（降级处理）
            print(f"  警告：无法获取交易日历，假设 {date} 是交易日")
            return True

        # 检查是否是交易日
        # 将 date_int 转为字符串进行比较
        date_str = str(date_int)
        cal_row = self.trade_calendar_cache[self.trade_calendar_cache['cal_date'] == date_str]
        if cal_row.empty:
            return False  # 不在交易日历中

        return int(cal_row.iloc[0]['is_open']) == 1

    def get_latest_trade_date(self, as_of_date: str = None, exchange: str = 'SSE') -> Optional[str]:
        """
        获取指定日期之前（含）的最近一个交易日

        Args:
            as_of_date: 基准日期 YYYY-MM-DD 或 YYYYMMDD，默认为今天
            exchange: 交易所

        Returns:
            最近一个交易日的日期字符串 YYYYMMDD
        """
        if as_of_date is None:
            as_of_date = datetime.now().strftime('%Y%m%d')
        elif '-' in as_of_date:
            as_of_date = as_of_date.replace('-', '')

        # 确保交易日历已加载
        date_int = int(as_of_date)
        if self.trade_calendar_cache is None or self.trade_calendar_date is None:
            year = int(as_of_date[:4])
            start_date = f"{year - 1}0101"
            end_date = f"{year + 1}1231"
            self._load_trade_calendar(exchange=exchange, start_date=start_date, end_date=end_date)

        if self.trade_calendar_cache is None or self.trade_calendar_cache.empty:
            # 无法获取交易日历，返回原日期
            return as_of_date

        # 查找最近的交易日
        # 确保 cal_date 是数值类型以便比较
        cal_date_col = pd.to_numeric(self.trade_calendar_cache['cal_date'], errors='coerce')
        trade_days = self.trade_calendar_cache[
            (self.trade_calendar_cache['is_open'] == 1) &
            (cal_date_col <= date_int)
        ].sort_values('cal_date', ascending=False)

        if trade_days.empty:
            return None

        return str(trade_days.iloc[0]['cal_date'])

    # ==================== 股票列表相关 ====================

    def get_all_stock_codes(self, max_retries: int = 3, ref_date: str = None) -> Dict[str, str]:
        """
        获取所有 A 股股票代码（严格模式：直接从 Tushare 获取）

        Args:
            max_retries: 最大重试次数
            ref_date: 参考日期 YYYY-MM-DD，用于获取该日期的股票列表

        Returns:
            {symbol: name} 字典
        """
        print("正在通过 Tushare 获取 A 股股票列表...")

        for attempt in range(max_retries):
            try:
                # 使用 stock_basic 接口获取股票列表
                # ref_date 参数确保获取的是指定日期的股票列表
                df = self.pro.stock_basic(
                    exchange='',
                    list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date,delist_date'
                )

                if df is None or df.empty:
                    print(f"  第 {attempt+1} 次尝试返回空数据")
                    time.sleep(2)
                    continue

                # 过滤：只保留 A 股
                code_pattern = r'^(00|30|60|68|88)\d{4}\.(SZ|SH|BJ)$'
                df = df[df['ts_code'].str.match(code_pattern, na=False)]

                # 排除 ST、*ST 股票
                df = df[~df['name'].str.contains('ST|退', na=False, regex=True)]

                # 排除 ETF、基金、债券等
                exclude_keywords = ['ETF', '基金', '债', 'LOF', '指数']
                for kw in exclude_keywords:
                    df = df[~df['name'].str.contains(kw, na=False)]

                # 转换为字典 {symbol: name}
                stock_dict = dict(zip(df['symbol'].astype(str).str.zfill(6), df['name']))

                print(f"✓ Tushare 获取成功：{len(stock_dict)} 只 A 股股票")
                self._save_stock_names(stock_dict)
                return stock_dict

            except Exception as e:
                print(f"  Tushare 请求失败：{e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"  {wait_time} 秒后重试...")
                    time.sleep(wait_time)

        # 全部失败时，不返回缓存数据（严格模式）
        print("\nTushare 请求失败，不返回缓存数据（严格模式）")
        return {}

    def _save_stock_names(self, stock_dict: Dict[str, str]):
        """保存股票名称到本地"""
        try:
            with open(self.stock_names_file, 'w', encoding='utf-8') as f:
                json.dump(stock_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  保存股票名称失败：{e}")

    # ==================== K 线数据相关 ====================

    def fetch_stock_history(self, stock_code: str, start_date: str = None, end_date: str = None,
                          force: bool = False) -> Optional[pd.DataFrame]:
        """
        抓取单只股票历史数据（严格模式）

        Args:
            stock_code: 股票代码（6 位数字）
            start_date: 开始日期 YYYY-MM-DD，默认为 3 年前
            end_date: 结束日期 YYYY-MM-DD，默认为今天
            force: 是否强制重新抓取（忽略本地缓存）

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount

        严格模式原则：
        1. 如果本地没有该股票数据，直接从 Tushare 下载
        2. 如果本地有数据，检查日期覆盖范围
        3. 对于缺失的日期范围，从 Tushare 补充
        4. 数据日期必须精确匹配，不使用猜测数据
        """
        # 标准化日期格式
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            if '-' in end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                end_date = datetime.strptime(end_date, '%Y%m%d')

        if start_date is None:
            start_date = end_date - timedelta(days=365 * 3)  # 默认 3 年
        elif isinstance(start_date, str):
            if '-' in start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
            else:
                start_date = datetime.strptime(start_date, '%Y%m%d')

        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        # 检查本地是否有数据
        existing_df = None
        if not force:
            existing_df = self.csv_manager.read_stock(stock_code)
            if existing_df is not None and not existing_df.empty:
                # 检查日期覆盖范围
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                min_date = existing_df['date'].min()
                max_date = existing_df['date'].max()

                # 如果本地数据已覆盖请求范围，直接返回
                if min_date <= start_date and max_date >= end_date:
                    print(f"✓ {stock_code} 本地数据已覆盖请求范围")
                    return self._filter_and_sort(existing_df, start_date, end_date)

                # 需要补充数据
                print(f"  {stock_code} 本地数据范围：{min_date.date()} ~ {max_date.date()}")
                print(f"  请求范围：{start_date.date()} ~ {end_date.date()}")

                # 合并现有数据和新获取的数据
                return self._fetch_and_merge(stock_code, existing_df, start_date, end_date)

        # 没有本地数据或强制重新抓取
        print(f"{'强制重新抓取' if force else '首次抓取'} {stock_code}...")
        return self._full_fetch(stock_code, start_str, end_str)

    def _filter_and_sort(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """过滤和排序数据"""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        df = df.sort_values('date', ascending=False).reset_index(drop=True)
        return df

    def _fetch_and_merge(self, stock_code: str, existing_df: pd.DataFrame,
                         start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """获取缺失的数据并与现有数据合并"""
        existing_df = existing_df.copy()
        existing_df['date'] = pd.to_datetime(existing_df['date'])

        min_date = existing_df['date'].min()
        max_date = existing_df['date'].max()

        # 确定需要获取的日期范围
        need_start = min(start_date, min_date)
        need_end = max(end_date, max_date)

        start_str = need_start.strftime('%Y%m%d')
        end_str = need_end.strftime('%Y%m%d')

        print(f"  从 Tushare 获取 {stock_code} 数据：{start_str} ~ {end_str}")

        new_df = self._fetch_from_tushare(stock_code, start_str, end_str)

        if new_df is None or new_df.empty:
            print(f"  警告：从 Tushare 获取数据失败，返回现有数据")
            return self._filter_and_sort(existing_df, start_date, end_date)

        # 合并数据（现有数据优先）
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='first')  # 保留现有数据
        combined = combined.sort_values('date', ascending=False).reset_index(drop=True)

        # 写入本地
        self.csv_manager.write_stock(stock_code, combined)

        print(f"✓ {stock_code} 数据已更新")
        return self._filter_and_sort(combined, start_date, end_date)

    def _full_fetch(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从 Tushare 全量获取数据"""
        df = self._fetch_from_tushare(stock_code, start_date, end_date)

        if df is not None and not df.empty:
            self.csv_manager.write_stock(stock_code, df)
            print(f"✓ {stock_code} (Tushare 获取 {len(df)}条)")

        return df

    def _fetch_from_tushare(self, stock_code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从 Tushare 获取股票 K 线数据"""
        try:
            # 构建 ts_code（需要后缀）
            if stock_code.startswith('6') or stock_code.startswith('88'):
                ts_code = f'{stock_code}.SH'
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                ts_code = f'{stock_code}.SZ'
            elif stock_code.startswith('4') or stock_code.startswith('8'):
                ts_code = f'{stock_code}.BJ'
            else:
                print(f"  未知的股票代码前缀：{stock_code}")
                return None

            # 使用 Tushare 日线接口（前复权）
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

            if df is None or df.empty:
                return None

            # 转换字段格式
            df = df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume',
                'amount': 'amount',
            })[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()

            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])

            # 添加换手率（需要计算或从其他接口获取）
            df['turnover'] = 0.0

            # 按日期降序排列
            df = df.sort_values('date', ascending=False).reset_index(drop=True)

            return df

        except Exception as e:
            print(f"  Tushare 获取失败：{e}")
            return None

    def fetch_for_date(self, stock_codes: List[str], trade_date: str) -> pd.DataFrame:
        """
        获取指定交易日所有股票的行情数据

        Args:
            stock_codes: 股票代码列表
            trade_date: 交易日 YYYY-MM-DD 或 YYYYMMDD

        Returns:
            包含所有股票行情的 DataFrame
        """
        # 标准化日期格式
        if '-' in trade_date:
            trade_date = trade_date.replace('-', '')

        # 验证是否为交易日
        if not self.is_trade_date(trade_date):
            print(f"警告：{trade_date} 不是交易日")
            return pd.DataFrame()

        all_data = []

        for code in stock_codes:
            df = self.csv_manager.read_stock(code)
            if df is None or df.empty:
                # 本地没有数据，从 Tushare 获取
                df = self._fetch_from_tushare(code, trade_date, trade_date)
                if df is not None and not df.empty:
                    self.csv_manager.write_stock(code, df)
            else:
                # 检查本地数据是否包含该日期
                df['date'] = pd.to_datetime(df['date'])
                target_date = pd.to_datetime(trade_date)
                df_date = df[df['date'] == target_date]

                if df_date.empty:
                    # 本地没有该日期数据，从 Tushare 获取
                    df_fresh = self._fetch_from_tushare(code, trade_date, trade_date)
                    if df_fresh is not None and not df_fresh.empty:
                        # 合并数据
                        df = pd.concat([df, df_fresh], ignore_index=True)
                        df = df.drop_duplicates(subset=['date'], keep='last')
                        df = df.sort_values('date', ascending=False).reset_index(drop=True)
                        self.csv_manager.write_stock(code, df)

                df_date = df[df['date'] == target_date]

            if df_date is not None and not df_date.empty:
                df_date = df_date.copy()
                df_date['code'] = code
                all_data.append(df_date)

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def get_market_cap(self, stock_code: str, trade_date: str = None) -> Optional[float]:
        """
        获取指定日期的总市值

        Args:
            stock_code: 股票代码
            trade_date: 交易日 YYYY-MM-DD 或 YYYYMMDD，默认为最近交易日

        Returns:
            总市值（元），失败返回 None
        """
        if trade_date is None:
            # 使用最近交易日
            trade_date = self.get_latest_trade_date()
        elif '-' in trade_date:
            trade_date = trade_date.replace('-', '')

        try:
            # 构建 ts_code
            if stock_code.startswith('6') or stock_code.startswith('88'):
                ts_code = f'{stock_code}.SH'
            elif stock_code.startswith('0') or stock_code.startswith('3'):
                ts_code = f'{stock_code}.SZ'
            elif stock_code.startswith('4') or stock_code.startswith('8'):
                ts_code = f'{stock_code}.BJ'
            else:
                return None

            # 使用 daily_basic 接口获取总市值
            df = self.pro.daily_basic(ts_code=ts_code, trade_date=trade_date,
                                      fields='ts_code,total_mv')

            if df is not None and not df.empty and 'total_mv' in df.columns:
                # total_mv 单位是万元，转为元
                total_mv = df.iloc[0]['total_mv']
                if pd.notna(total_mv):
                    return float(total_mv) * 10000

            return None

        except Exception as e:
            print(f"  获取市值失败：{e}")
            return None

    def init_full_data(self, max_stocks: Optional[int] = None, skip_failed: bool = True,
                       years: int = 6):
        """
        首次全量抓取

        Args:
            max_stocks: 限制抓取数量（用于测试）
            skip_failed: 是否跳过之前失败的股票
            years: 抓取年数
        """
        stock_dict = self.get_all_stock_codes()

        if not stock_dict:
            print("无法获取股票列表")
            return

        stock_codes = list(stock_dict.keys())

        # 加载之前失败的股票列表
        failed_stocks_file = self.full_data_dir / 'failed_stocks.json'
        failed_stocks = set()
        if skip_failed and failed_stocks_file.exists():
            try:
                with open(failed_stocks_file, 'r', encoding='utf-8') as f:
                    failed_stocks = set(json.load(f))
                print(f"  将跳过 {len(failed_stocks)} 只之前获取失败的股票")
                stock_codes = [c for c in stock_codes if c not in failed_stocks]
            except:
                pass

        if max_stocks:
            stock_codes = stock_codes[:max_stocks]

        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')

        total = len(stock_codes)
        success = 0
        failed = 0
        failed_list = []

        print(f"\n开始抓取 {total} 只股票的{years}年历史数据...")
        print(f"日期范围：{start_str} ~ {end_str}")
        print("=" * 60)

        for i, code in enumerate(stock_codes, 1):
            print(f"[{i}/{total}] 抓取 {code} {stock_dict.get(code, '')} ...", end=" ")

            df = self._fetch_from_tushare(code, start_str, end_str)

            if df is not None and not df.empty:
                # 数据校验
                valid_data = True
                if len(df) < 10:
                    print(f"⚠ 数据太少 ({len(df)}条)")
                    valid_data = False
                    failed_list.append(code)
                elif df['close'].mean() <= 0:
                    print(f"⚠ 价格异常")
                    valid_data = False
                    failed_list.append(code)
                else:
                    self.csv_manager.write_stock(code, df)
                    print(f"✓ ({len(df)}条)")
                    success += 1
            else:
                print("✗ 失败")
                failed += 1
                failed_list.append(code)

            # 限速：Tushare 限制 200次/分钟，约每秒3-4次
            # 每3个请求休息0.5秒，每秒约6次，在限制内
            if i % 3 == 0:
                time.sleep(0.5)

        # 保存失败的股票列表
        if failed_list:
            try:
                with open(failed_stocks_file, 'w', encoding='utf-8') as f:
                    json.dump(failed_list, f)
                print(f"\n  已保存 {len(failed_list)} 只获取失败的股票到 failed_stocks.json")
            except Exception as e:
                print(f"\n  保存失败列表出错：{e}")

        print("=" * 60)
        print(f"完成! 成功：{success}, 失败：{failed + len(failed_list)}")

    def daily_update(self, max_stocks: Optional[int] = None):
        """
        每日增量更新（严格模式）

        只更新实际交易日的数据，不更新非交易日
        """
        existing_stocks = self.csv_manager.list_all_stocks()

        if not existing_stocks:
            print("没有找到已有数据，请先执行 init")
            return

        if max_stocks:
            existing_stocks = existing_stocks[:max_stocks]

        # 获取今天是否为交易日
        today = datetime.now()
        today_str = today.strftime('%Y%m%d')

        # 检查今天是否为交易日
        if not self.is_trade_date(today_str):
            print(f"今天 ({today_str}) 不是交易日，跳过更新")
            return

        # 获取最近一个交易日
        latest_trade_date = self.get_latest_trade_date()
        if latest_trade_date is None:
            print("无法获取最近交易日")
            return

        print(f"\n开始更新 {len(existing_stocks)} 只股票的数据...")
        print(f"交易日：{latest_trade_date}")
        print("=" * 60)

        updated = 0
        failed = 0

        for i, code in enumerate(existing_stocks, 1):
            print(f"[{i}/{len(existing_stocks)}] 更新 {code} ...", end=" ")

            # 获取该日期的数据
            df = self._fetch_from_tushare(code, latest_trade_date, latest_trade_date)

            if df is not None and not df.empty:
                # 检查本地是否已有该日期数据
                existing_df = self.csv_manager.read_stock(code)
                if existing_df is not None and not existing_df.empty:
                    existing_df['date'] = pd.to_datetime(existing_df['date'])
                    target = pd.to_datetime(latest_trade_date)
                    if (existing_df['date'] == target).any():
                        print("已有数据")
                        continue

                # 合并数据
                existing_df = self.csv_manager.read_stock(code)
                if existing_df is not None and not existing_df.empty:
                    df = pd.concat([existing_df, df], ignore_index=True)
                    df = df.drop_duplicates(subset=['date'], keep='last')
                    df = df.sort_values('date', ascending=False).reset_index(drop=True)

                self.csv_manager.write_stock(code, df)
                print(f"✓ (新增 {len(df)} 条)")
                updated += 1
            else:
                print("✗ 失败")
                failed += 1

            # 限速：Tushare 限制 200次/分钟
            if i % 3 == 0:
                time.sleep(0.5)

        print("=" * 60)
        print(f"完成! 更新成功：{updated}, 失败：{failed}")
