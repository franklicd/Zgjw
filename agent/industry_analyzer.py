"""
industry_analyzer.py
~~~~~~~~~~~~~~~~~~~~
股票行业数据分析模块

功能：
    1. 获取股票所属行业信息
    2. 计算行业热度（成交额占比）
    3. 提供行业数据用于推荐股票分析

数据来源（多数据源轮询）：
    1. 东方财富行业板块（akshare）
    2. 腾讯财经行业数据
    3. 新浪财经行业数据
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None

import requests

try:
    from .industry_fetcher import IndustryFetcher
except ImportError:
    from industry_fetcher import IndustryFetcher


class IndustryAnalyzer:
    """行业数据分析器"""

    def __init__(self, data_dir: str | Path | None = None):
        """
        初始化行业分析器

        Args:
            data_dir: 数据缓存目录（可选）
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._industry_cache: Dict[str, str] = {}  # code -> industry
        self._market_data_cache: Optional[pd.DataFrame] = None

        # 初始化行业数据获取器（带缓存）
        cache_dir = self.data_dir / "industry_cache" if self.data_dir else None
        self.fetcher = IndustryFetcher(cache_dir=cache_dir)
        # 自动加载行业映射（缓存优先，180 天有效期）
        self.fetcher.load_industry_mapping(cache_days=180)

    def load_industry_from_stocklist(self, stocklist_path: str | Path) -> Dict[str, str]:
        """
        从 stocklist.csv 加载行业信息

        Args:
            stocklist_path: stocklist.csv 文件路径

        Returns:
            {code: industry} 字典
        """
        stocklist_path = Path(stocklist_path)
        if not stocklist_path.exists():
            return {}

        df = pd.read_csv(stocklist_path)
        if 'industry' not in df.columns:
            return {}

        # 建立 code -> industry 映射（确保代码为 6 位字符串格式）
        self._industry_cache = dict(zip(df['symbol'].astype(str).str.zfill(6), df['industry']))
        return self._industry_cache

    def get_stock_industry(self, code: str) -> Optional[str]:
        """
        获取股票所属行业

        Args:
            code: 股票代码（6 位数字）

        Returns:
            行业名称，如果找不到返回 None
        """
        code = str(code).zfill(6)
        if code in self._industry_cache:
            return self._industry_cache[code]
        # 尝试从 IndustryFetcher 获取（带缓存）
        return self.fetcher.get_stock_industry(code)

    def fetch_industry_turnover(self, max_retries=5) -> Optional[pd.DataFrame]:
        """
        获取行业成交额数据（多数据源轮询）

        数据源优先级：
        1. 东方财富行业板块（akshare）- 提供真实成交额数据
        2. 腾讯财经行业数据
        3. 新浪财经行业数据 - 仅提供行业名称，成交额为 0

        Args:
            max_retries: 每个数据源的最大重试次数

        Returns:
            DataFrame with columns: industry, turnover, change_pct, ...
        """
        # 数据源列表
        data_sources = [
            ("东方财富 (AKShare)", self._fetch_from_akshare, True),  # True=有真实成交额数据
            ("腾讯财经", self._fetch_from_tencent, False),
            ("新浪财经", self._fetch_from_sina, False),
        ]

        for source_name, fetch_func, has_real_turnover in data_sources:
            print(f"[INFO] 尝试从 {source_name} 获取行业数据...")
            for attempt in range(max_retries):
                try:
                    df = fetch_func()
                    if df is not None and not df.empty:
                        # 如果没有真实成交额数据，添加警告
                        if not has_real_turnover:
                            print(f"[WARN] {source_name} 仅提供行业名称，成交额数据为估算值")
                        print(f"[INFO] 成功从 {source_name} 获取行业数据，共 {len(df)} 条记录")
                        return df
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 1.0
                        print(f"[WARN] {source_name} 获取失败 ({e})，{wait_time}秒后重试 ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"[WARN] {source_name} 获取失败，已重试{max_retries}次")

        print("[ERROR] 所有数据源均获取失败，返回空数据")
        return None

    def _fetch_from_akshare(self) -> Optional[pd.DataFrame]:
        """从东方财富获取行业数据"""
        if ak is None:
            raise ImportError("akshare 未安装")

        # 获取行业板块数据（东方财富）
        df = ak.stock_board_industry_name_em()

        # 标准化列名
        df = df.rename(columns={
            '板块名称': 'industry',
            '成交额': 'turnover',
            '涨跌幅': 'change_pct',
            '板块股票数量': 'stock_count',
        })

        return df[['industry', 'turnover', 'change_pct', 'stock_count']].copy()

    def _fetch_from_tencent(self) -> Optional[pd.DataFrame]:
        """
        从腾讯财经获取行业数据

        腾讯财经 API：http://web.ifzq.gtimg.cn/appstock/app/fzkline/get
        备用：http://data.gtimg.com/
        """
        # 腾讯财经行业列表 API
        url = "http://data.gtimg.com/funddata/hy.js"

        headers = {
            "Referer": "http://data.gtimg.com/funddata/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 解析 Javascript 数据
        # 格式：var hqlist="银行|80700|100|基金|000001|100|..."
        content = response.text.strip()
        if not content.startswith('var hqlist="'):
            raise ValueError("腾讯财经返回数据格式异常")

        content = content.replace('var hqlist="', '').rstrip('";')

        # 解析行业列表
        industries = []
        parts = content.split('|')

        i = 0
        while i < len(parts) - 2:
            name = parts[i]
            code = parts[i + 1]
            # 第三个是市场代码，跳过
            if name and code:
                industries.append({
                    'industry': name,
                    'code': code,
                })
            i += 3

        if not industries:
            raise ValueError("腾讯财经解析后行业列表为空")

        # 获取每个行业的成交额（需要进一步请求）
        # 简化处理：返回基础行业列表，成交额设为估算值
        df = pd.DataFrame(industries)

        # 添加估算的成交额数据（按行业典型规模）
        # 这里只是占位，实际应该调用更多 API 获取真实数据
        df['turnover'] = 0.0  # 暂时设为 0
        df['change_pct'] = 0.0
        df['stock_count'] = 0

        return df[['industry', 'turnover', 'change_pct', 'stock_count']].copy()

    def _fetch_from_sina(self) -> Optional[pd.DataFrame]:
        """
        从新浪财经获取行业数据

        新浪财经行业 API：http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodes
        数据格式：["分类 1", [市场列表], "分类 3", "分类 4"]
        市场列表：[['A 股', [['新浪行业', [['玻璃行业', '', 'new_blhy'], ...]], ...], '', 'sinahy', 'cn'], ...]
        """
        url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodes"

        headers = {
            "Referer": "http://vip.stock.finance.sina.com.cn/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 解析 JSON 数据
        data = response.json()
        industries = []

        # 数据结构：["分类 1", [市场列表], "分类 3", "分类 4"]
        # 市场列表：[['A 股', [行业分类列表], '', 'sinahy', 'cn'], ...]
        if isinstance(data, list) and len(data) > 1:
            market_list = data[1]  # 市场列表
            if isinstance(market_list, list):
                for market in market_list:
                    if isinstance(market, list) and len(market) > 1:
                        # market 格式：['A 股', [行业分类列表], '', 'sinahy', 'cn']
                        # market[0] = 市场名（如'A 股'）
                        # market[1] = 行业分类列表 [[行业分类 1], [行业分类 2], ...]
                        industry_categories = market[1]
                        if isinstance(industry_categories, list):
                            self._parse_sina_industry_categories(industry_categories, industries)

        if not industries:
            raise ValueError("新浪财经解析后行业列表为空")

        df = pd.DataFrame(industries)
        return df[['industry', 'turnover', 'change_pct', 'stock_count']].copy()

    def _parse_sina_industry_categories(self, categories: list, industries: list):
        """解析新浪行业分类列表"""
        for category in categories:
            if isinstance(category, list) and len(category) > 1:
                category_name = category[0]  # e.g., '新浪行业'
                industry_list = category[1]  # e.g., [['玻璃行业', '', 'new_blhy'], ...]

                if isinstance(industry_list, list):
                    for ind in industry_list:
                        if isinstance(ind, list) and len(ind) > 0:
                            industry_name = ind[0]
                            # 过滤掉分类标签，只保留具体行业
                            if industry_name and isinstance(industry_name, str) and \
                               industry_name not in ['新浪行业', '申万行业', '申万一级', '申万二级', '申万三级',
                                                      '热门概念', '概念板块', '地域板块', '证监会行业', '分类']:
                                industries.append({
                                    'industry': industry_name,
                                    'turnover': 0.0,
                                    'change_pct': 0.0,
                                    'stock_count': 0,
                                })

    def fetch_market_total_turnover(self) -> Optional[float]:
        """
        获取全市场总成交额

        Returns:
            全市场总成交额（亿元），失败返回 None
        """
        if ak is None:
            return None

        try:
            # 获取沪深两市成交额
            sh_df = ak.stock_sh_a_spot_em()  # 沪 A
            sz_df = ak.stock_sz_a_spot_em()  # 深 A

            sh_turnover = sh_df['成交额'].sum() / 1e8  # 转为亿元
            sz_turnover = sz_df['成交额'].sum() / 1e8

            return sh_turnover + sz_turnover
        except Exception as e:
            print(f"[WARN] 获取市场总成交额失败：{e}")
            return None

    def calculate_industry_heat(self, pick_date: str | None = None) -> Dict[str, Dict]:
        """
        计算行业热度（成交额占比）

        优先使用外部数据源（东方财富/腾讯/新浪）获取行业成交额数据
        如果外部数据源不可用，则使用候选股票的交易额数据计算行业热度

        Args:
            pick_date: 选股日期（YYYY-MM-DD），用于从候选数据中计算

        Returns:
            {
                industry_name: {
                    'turnover': 成交额（亿元）,
                    'change_pct': 涨跌幅，
                    'market_ratio': 占比（%）,
                    'stock_count': 股票数量
                }
            }
        """
        # 尝试获取外部行业数据
        industry_df = self.fetch_industry_turnover()

        # 如果外部数据不可用或没有成交额数据，使用候选股票数据计算
        if industry_df is None or industry_df.empty or (industry_df is not None and 'turnover' in industry_df.columns and industry_df['turnover'].sum() == 0):
            print("[INFO] 外部数据源不可用，尝试从候选股票数据计算行业热度...")
            return self._calculate_industry_heat_from_candidates(pick_date)

        # 统一列名映射（支持不同数据源的列名）
        # 东方财富列名：'板块名称', '成交额', '涨跌幅', '板块股票数量'
        # 标准化列名：'industry', 'turnover', 'change_pct', 'stock_count'
        df = industry_df.copy()
        if '板块名称' in df.columns:
            df = df.rename(columns={'板块名称': 'industry', '成交额': 'turnover', '涨跌幅': 'change_pct', '板块股票数量': 'stock_count'})

        # 计算总成交额
        total_turnover = float(df['turnover'].sum()) if 'turnover' in df.columns else 0.0

        result = {}
        for _, row in df.iterrows():
            industry_name = str(row.get('industry', '未知'))
            turnover = float(row.get('turnover', 0)) / 1e8  # 转为亿元

            result[industry_name] = {
                'turnover': round(turnover, 2),
                'change_pct': float(row.get('change_pct', 0)),
                'market_ratio': round((turnover / total_turnover * 100) if total_turnover > 0 else 0, 2),
                'stock_count': int(row.get('stock_count', 0)),
            }

        return result

    def _calculate_industry_heat_from_candidates(self, pick_date: str | None = None) -> Dict[str, Dict]:
        """
        从候选股票数据计算行业热度（备用方案）

        通过读取候选股票列表和对应的交易额数据，按行业分组计算：
        - 每个行业的总交易额
        - 该行业交易额占所有候选股票总交易额的比例

        Args:
            pick_date: 选股日期（YYYY-MM-DD）

        Returns:
            {
                industry_name: {
                    'turnover': 成交额（亿元）,
                    'change_pct': 涨跌幅（暂设为 0）,
                    'market_ratio': 占比（%）,
                    'stock_count': 股票数量
                }
            }
        """
        # 从 stocklist 获取行业信息和交易额
        from pathlib import Path

        # 候选股票文件
        candidates_paths = [
            Path(__file__).parent.parent / "data" / "candidates" / "candidates_latest.json",
        ]

        candidates_data = None
        for cp in candidates_paths:
            if cp.exists():
                with open(cp, encoding='utf-8') as f:
                    candidates_data = json.load(f)
                break

        if not candidates_data:
            print("[WARN] 无法加载候选股票数据，行业热度计算失败")
            return {}

        candidates = candidates_data.get("candidates", [])
        if not candidates:
            print("[WARN] 候选股票列表为空")
            return {}

        # 按行业分组统计交易额
        industry_turnover = {}
        industry_stocks = {}

        for stock in candidates:
            code = stock.get("code", "")
            turnover_n = stock.get("turnover_n", 0)  # 交易额（元）

            # 获取行业
            industry = self.get_stock_industry(code)
            if not industry:
                industry = "未知"

            # 累加行业交易额
            if industry not in industry_turnover:
                industry_turnover[industry] = 0
                industry_stocks[industry] = []
            industry_turnover[industry] += turnover_n
            industry_stocks[industry].append(code)

        # 计算总交易额
        total_turnover = sum(industry_turnover.values())

        # 构建结果
        result = {}
        for industry, turnover in industry_turnover.items():
            turnover_亿元 = turnover / 1e8  # 转为亿元
            result[industry] = {
                'turnover': round(turnover_亿元, 2),
                'change_pct': 0.0,  # 候选数据中无涨跌幅信息
                'market_ratio': round((turnover / total_turnover * 100) if total_turnover > 0 else 0, 2),
                'stock_count': len(industry_stocks.get(industry, [])),
            }

        print(f"[INFO] 从候选股票计算行业热度：共 {len(result)} 个行业，候选股票总成交额 {total_turnover/1e8:.2f} 亿元")
        return result

    def get_industry_rank(self, industry_name: str, industry_heat: Dict) -> Tuple[int, int]:
        """
        获取行业在全部行业中的排名

        Args:
            industry_name: 行业名称
            industry_heat: 行业热度字典

        Returns:
            (rank, total) 排名和总数
        """
        if not industry_heat or industry_name not in industry_heat:
            return 0, 0

        # 按成交额排序
        sorted_industries = sorted(
            industry_heat.items(),
            key=lambda x: x[1].get('turnover', 0),
            reverse=True
        )

        for i, (name, _) in enumerate(sorted_industries):
            if name == industry_name:
                return i + 1, len(sorted_industries)

        return 0, len(sorted_industries)

    def save_industry_heat(self, output_path: str | Path, industry_heat: Dict):
        """保存行业热度数据为 JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(industry_heat, f, ensure_ascii=False, indent=2)

    def load_industry_heat(self, input_path: str | Path) -> Dict:
        """从 JSON 文件加载行业热度数据"""
        input_path = Path(input_path)
        if not input_path.exists():
            return {}
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


# ────────────────────────────────────────────────
# 快速测试
# ────────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = IndustryAnalyzer()

    # 从 stocklist 加载行业
    stocklist_path = Path(__file__).parent.parent / "pipeline" / "stocklist.csv"
    if stocklist_path.exists():
        analyzer.load_industry_from_stocklist(stocklist_path)
        print(f"已加载 {len(analyzer._industry_cache)} 只股票的行业信息")

    # 获取行业热度
    print("\n计算行业热度...")
    industry_heat = analyzer.calculate_industry_heat()

    if industry_heat:
        print(f"共 {len(industry_heat)} 个行业")

        # 显示前 10 大行业
        sorted_heat = sorted(
            industry_heat.items(),
            key=lambda x: x[1].get('turnover', 0),
            reverse=True
        )[:10]

        print("\n前 10 大行业（按成交额）:")
        print(f"{'排名':>4} {'行业':>15} {'成交额 (亿)':>12} {'占比':>8} {'涨跌幅':>8}")
        print("-" * 55)
        for i, (name, data) in enumerate(sorted_heat, 1):
            print(f"{i:>4} {name:>15} {data['turnover']:>12.2f} {data['market_ratio']:>7.1f}% {data['change_pct']:>+7.2f}%")
