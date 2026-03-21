from __future__ import annotations

import datetime as dt
import logging
import random
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional
import os

import baostock as bs
import pandas as pd
import yaml
from tqdm import tqdm

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

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁", "请稍后", "超过频率", "频繁访问",
    "too many requests", "429",
    "forbidden", "403",
    "max retries exceeded"
)

def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)

class RateLimitError(RuntimeError):
    """表示命中限流/封禁，需要长时间冷却后重试。"""
    pass

def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)

# --------------------------- 历史K线（Baostock 日线，前复权） --------------------------- #

def _to_bs_code(code: str) -> str:
    """把6位code映射到baostock代码格式。"""
    code = str(code).zfill(6)
    if code.startswith(("60", "68", "9")):
        return f"sh.{code}"
    elif code.startswith(("4", "8")):
        return f"bj.{code}"
    else:
        return f"sz.{code}"

def _get_kline_baostock(code: str, start: str, end: str) -> pd.DataFrame:
    """
    通过 Baostock 获取 A 股日线前复权数据。
    start/end 格式：YYYYMMDD，内部转为 YYYY-MM-DD。
    """
    bs_code = _to_bs_code(code)
    # 转换日期格式 YYYYMMDD -> YYYY-MM-DD
    start_date = f"{start[:4]}-{start[4:6]}-{start[6:8]}"
    end_date = f"{end[:4]}-{end[4:6]}-{end[6:8]}"

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="1"  # 1=前复权
        )
        if rs.error_code != '0':
            logger.warning(f"{code} 获取数据失败: {rs.error_msg}")
            return pd.DataFrame()
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
    except Exception as e:
        if _looks_like_ip_ban(e):
            raise RateLimitError(str(e)) from e
        raise

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(columns={
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
    })[["date", "open", "close", "high", "low", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)

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

# --------------------------- 单只抓取（全量覆盖保存） --------------------------- #
def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
) -> bool:
    """抓取单只股票并保存。返回 True 表示成功，False 表示三次均失败。"""
    csv_path = out_dir / f"{code}.csv"

    for attempt in range(1, 4):
        try:
            new_df = _get_kline_baostock(code, start, end)
            if new_df.empty:
                logger.debug("%s 无数据，生成空表。", code)
                new_df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
            new_df = validate(new_df)
            new_df.to_csv(csv_path, index=False)  # 直接覆盖保存
            time.sleep(random.uniform(0.3, 0.6))  # 降低并发压力
            return True
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 3
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)

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
def main(log_path: Optional[Path] = None):
    # ---------- 读取 YAML 配置 ---------- #
    cfg = _load_config()

    # ---------- 日志路径（优先参数，其次 YAML，最后默认值） ---------- #
    if log_path is None:
        cfg_log = cfg.get("log")
        log_path = _resolve_cfg_path(cfg_log) if cfg_log else _default_log_path()
    setup_logging(log_path)
    logger.info("日志文件：%s", Path(log_path).resolve())

    # ---------- Baostock 登录 ---------- #
    logger.info("正在登录 Baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        logger.error(f"Baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    logger.info("Baostock 登录成功")

    # ---------- 日期解析 ---------- #
    raw_start = str(cfg.get("start", "20190101"))
    raw_end   = str(cfg.get("end",   "today"))
    start = dt.date.today().strftime("%Y%m%d") if raw_start.lower() == "today" else raw_start
    end   = dt.date.today().strftime("%Y%m%d") if raw_end.lower()   == "today" else raw_end

    out_dir = _resolve_cfg_path(cfg.get("out", "./data"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    stocklist_path = _resolve_cfg_path(cfg.get("stocklist", "./pipeline/stocklist.csv"))
    exclude_boards = set(cfg.get("exclude_boards") or [])
    codes = load_codes_from_stocklist(stocklist_path, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        bs.logout()
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:Baostock(日线,qfq) | 日期:%s → %s | 排除:%s",
        len(codes), start, end, ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 单线程抓取（全量覆盖） ---------- #
    ok_count   = 0
    fail_count = 0
    with tqdm(
        total=len(codes),
        desc="抓取进度",
        unit="支",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for code in codes:
            success = fetch_one(code, start, end, out_dir)
            if success:
                ok_count += 1
            else:
                fail_count += 1
            pbar.set_postfix(成功=ok_count, 失败=fail_count)
            pbar.update(1)

    # ---------- Baostock 登出 ---------- #
    bs.logout()
    logger.info("Baostock 已登出")

    logger.info("全部任务完成：成功 %d 支，失败 %d 支，数据已保存至 %s",
                ok_count, fail_count, out_dir.resolve())

if __name__ == "__main__":
    main()
