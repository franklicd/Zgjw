from __future__ import annotations

import datetime as dt
from datetime import timedelta
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

# 导入数据抓取模块（使用 Tushare）
from pipeline.tushare_fetcher import TushareFetcher
from pipeline.csv_manager import CSVManager

warnings.filterwarnings("ignore")

# --------------------------- 全局日志配置 --------------------------- #
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DIR = _PROJECT_ROOT / "data" / "logs"

def _resolve_cfg_path(path_like: str | Path, base_dir: Path = _PROJECT_ROOT) -> Path:
    """将配置中的路径统一解析为绝对路径：相对路径基于项目根目录。"""
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)

def _default_log_path() -> Path:
    today = dt.date.today().strftime("%Y-%m-%d")
    return _DEFAULT_LOG_DIR / f"fetch_{today}.log"

def setup_logging(log_path: Optional[Path] = None) -> None:
    """初始化日志：同时输出到 stdout 和指定文件。"""
    if log_path is None:
        log_path = _default_log_path()
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )

logger = logging.getLogger("fetch_from_stocklist")

# --------------------------- 历史 K 线（Tushare Pro 接口，前复权） --------------------------- #

def _get_kline_tushare(code: str, years: int = 6) -> pd.DataFrame:
    """
    通过 Tushare Pro 接口获取 A 股日线前复权数据。
    :param code: 6 位股票代码
    :param years: 获取最近几年的数据，默认 6 年
    """
    try:
        # 初始化 fetcher（单例模式，避免重复初始化）
        if not hasattr(_get_kline_tushare, "_fetcher"):
            _get_kline_tushare._fetcher = TushareFetcher(data_dir=str(_PROJECT_ROOT / "data"))

        # 计算日期范围
        from datetime import timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * years)

        df = _get_kline_tushare._fetcher.fetch_stock_history(
            code,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 统一字段格式，和原有系统兼容
        df = df.rename(columns={
            "date": "date",
            "open": "open",
            "close": "close",
            "high": "high",
            "low": "low",
            "volume": "volume",
        })[["date", "open", "close", "high", "low", "volume"]].copy()

        # 按日期正序排列（和原有 baostock 返回格式一致）
        df = df.sort_values("date").reset_index(drop=True)

        return df
    except Exception as e:
        raise

def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df

# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #

def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    ts = df["ts_code"].astype(str).str.upper()
    num = ts.str.extract(r"(\d{6})", expand=False).str.zfill(6)
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~((ts.str.endswith(".SZ")) & num.str.startswith(("300", "301")))
    if "star" in exclude_boards:
        mask &= ~((ts.str.endswith(".SH")) & num.str.startswith(("688",)))
    if "bj" in exclude_boards:
        mask &= ~((ts.str.endswith(".BJ")) | num.str.startswith(("4", "8")))

    return df[mask].copy()


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> List[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info("从 %s 读取到 %d 只股票（排除板块：%s）",
                stocklist_csv, len(codes), ",".join(sorted(exclude_boards)) or "无")
    return codes

# --------------------------- 单只抓取（支持增量更新） --------------------------- #

# 全局目标交易日（由 main() 设置）
_target_trade_date: Optional[str] = None

def fetch_one(
    code: str,
    years: int,
    out_dir: Path,
    force: bool = False,
    target_trade_date: Optional[str] = None,
) -> bool:
    """
    抓取单只股票并保存（支持增量更新）
    :param code: 股票代码
    :param years: 最大保留年数（默认 3 年）
    :param out_dir: 输出目录
    :param force: 是否强制重新抓取
    :param target_trade_date: 目标交易日 YYYYMMDD，如果提供则先检查是否已有该日期数据
    :return: True 表示成功，False 表示失败
    """
    # 使用 TushareFetcher 的增量更新功能
    if not hasattr(fetch_one, "_fetcher"):
        fetch_one._fetcher = TushareFetcher(data_dir=str(out_dir))

    fetcher = fetch_one._fetcher

    # 如果指定了目标交易日，先检查本地是否已有该日期数据
    if target_trade_date and not force:
        existing_df = fetcher.csv_manager.read_stock(code)
        if existing_df is not None and not existing_df.empty:
            existing_df['date'] = pd.to_datetime(existing_df['date'])
            target_date = pd.to_datetime(target_trade_date)
            if (existing_df['date'] == target_date).any():
                logger.debug("%s 已包含目标交易日 (%s) 数据，跳过", code, target_trade_date)
                return True  # 已有数据，跳过

    # 计算日期范围
    end_date = dt.datetime.now()
    start_date = end_date - timedelta(days=365 * years)

    for attempt in range(1, 4):
        try:
            df = fetcher.fetch_stock_history(
                code,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                force=force
            )
            if df is None or df.empty:
                logger.debug("%s 无数据，生成空表。", code)
                df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
                # 写入空表
                fetcher.csv_manager.write_stock(code, df)

            # Tushare 有调用频率限制，每次请求后稍作等待
            time.sleep(0.2)
            return True
        except Exception as e:
            # Tushare 错误通常是积分不足或调用频率超限
            logger.error(f"{code} 第 {attempt} 次抓取失败：{e}")
            if attempt < 3:
                time.sleep(2)  # 失败后等待再重试

    logger.error("%s 三次抓取均失败，已跳过！", code)
    return False




# --------------------------- 配置加载 --------------------------- #
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "fetch_kline.yaml"

def _load_config(config_path: Path = _CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info("已加载配置文件：%s", config_path.resolve())
    return cfg


# --------------------------- 主入口 --------------------------- #
def main(log_path: Optional[Path] = None, force: bool = False):
    """
    主函数
    :param log_path: 日志文件路径
    :param force: 是否强制重新抓取所有股票
    """
    # ---------- 读取 YAML 配置 ---------- #
    cfg = _load_config()

    # ---------- 日志路径（优先参数，其次 YAML，最后默认值） ---------- #
    if log_path is None:
        cfg_log = cfg.get("log")
        log_path = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    setup_logging(log_path)
    logger.info("日志文件：%s", Path(log_path).resolve())

    # ---------- 日期解析（默认改为 3 年） ---------- #
    raw_years = cfg.get("years", 3)
    try:
        years = int(raw_years)
    except:
        years = 3

    out_dir = _resolve_cfg_path(cfg.get("out", "./data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 检查交易日并确定选股日期 ---------- #
    # 初始化 fetcher 用于交易日判断
    fetcher = TushareFetcher(data_dir=str(out_dir))

    # 获取系统当前日期
    today = dt.date.today()
    today_str = today.strftime('%Y%m%d')

    # 判断今天是否为交易日
    if fetcher.is_trade_date(today_str):
        # 今天是交易日，目标日期就是今天
        target_trade_date = today_str
        logger.info("今天 (%s) 是交易日，将拉取最新数据", today_str)
    else:
        # 今天不是交易日，获取最近一个交易日
        latest_trade_date = fetcher.get_latest_trade_date()
        if latest_trade_date is None:
            logger.error("无法获取最近交易日")
            sys.exit(1)
        target_trade_date = latest_trade_date
        logger.info("今天 (%s) 不是交易日，最近交易日为 %s", today_str, latest_trade_date)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./pipeline/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:Tushare Pro (日线，qfq) | 最近:%d年 | 排除:%s | 强制刷新:%s",
        len(codes), years, ",".join(sorted(exclude_boards)) or "无", "是" if force else "否",
    )

    # ---------- 单线程抓取（支持增量更新） ---------- #
    ok_count   = 0
    fail_count = 0
    with tqdm(
        total=len(codes),
        desc="抓取进度",
        unit="支",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for code in codes:
            # 传入目标交易日，让 fetch_one 检查是否已有该日期数据
            success = fetch_one(code, years, out_dir, force=force, target_trade_date=target_trade_date)
            if success:
                ok_count += 1
            else:
                fail_count += 1
            pbar.set_postfix(成功=ok_count, 失败=fail_count)
            pbar.update(1)

    logger.info("全部任务完成：成功 %d 支，失败 %d 支，数据已保存至 %s",
                ok_count, fail_count, out_dir.resolve())

if __name__ == "__main__":
    main()
