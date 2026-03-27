# 数据缓存机制（严格模式）

## 核心原则

1. **如果选择日期下没有数据，不管是行情还是股票列表，则直接从 Tushare 下载**
2. **如果日期是交易日，则数据与日期必须要匹配**
3. **在计算任何统计值的时候，不可以猜测或用不匹配日期的缓存数据来代替实际的数据**
4. **数据的精确性排在第一位**

## 缓存架构

### 1. 交易日历缓存

**位置**: `data/trade_calendar_cache.parquet`

**用途**:
- 验证给定日期是否为交易日
- 获取最近一个交易日日期
- 确保数据日期精确匹配

**加载逻辑**:
```python
# 优先从 Tushare 加载交易日历
df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)

# 保存到本地 parquet 缓存
df.to_parquet('data/trade_calendar_cache.parquet')
```

**关键方法**:
- `is_trade_date(date)` - 检查是否为交易日
- `get_latest_trade_date(as_of_date)` - 获取最近交易日

### 2. 股票数据缓存

**位置**: `data/XX/{code}.parquet` (按股票代码前两位分目录)

**用途**: 存储股票历史 K 线数据

**缓存验证逻辑**:
```python
def fetch_stock_history(code, start_date, end_date, force=False):
    # 1. 读取本地缓存
    existing_df = csv_manager.read_stock(code)

    # 2. 检查日期覆盖范围
    min_date = existing_df['date'].min()
    max_date = existing_df['date'].max()

    # 3. 如果已覆盖请求范围，直接返回
    if min_date <= start_date and max_date >= end_date:
        return filter_and_sort(existing_df, start_date, end_date)

    # 4. 否则从 Tushare 补充缺失数据
    return fetch_and_merge(code, existing_df, start_date, end_date)
```

**关键特性**:
- 不依赖"今天已更新"的时间缓存判断
- 基于实际日期范围判断是否需要获取数据
- 数据不足时自动从 Tushare 补充

### 3. 股票列表缓存

**位置**: `data/stock_names.json`

**用途**: 缓存 A 股股票代码和名称映射

**严格模式行为**:
- 优先从 Tushare 直接获取最新股票列表
- 仅在 API 失败时返回空字典（不使用缓存）
- 避免使用过期的股票列表

## 数据流程图

```
用户请求数据 (code, date_range)
         │
         ▼
┌─────────────────────┐
│  1. 检查本地缓存    │
│     csv_manager     │
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
  有缓存      无缓存
    │           │
    ▼           ▼
┌─────────┐  ┌──────────────┐
│日期范围 │  │ 从 Tushare   │
│  检查   │  │ 全量获取     │
└────┬────┘  └──────────────┘
     │
  ┌──┴──────────────┐
  │                 │
已覆盖          未覆盖/部分覆盖
  │                 │
  ▼                 ▼
返回数据    ┌──────────────────┐
           │ 从 Tushare 获取   │
           │ 缺失日期范围数据  │
           └─────────┬────────┘
                     │
                     ▼
               ┌─────────────┐
               │ 合并数据    │
               │ (现有优先)  │
               └─────────────┘
```

## 使用示例

### 全量初始化
```bash
python -m pipeline.fetch_kline
```

### 增量更新
```bash
# 自动检测缺失数据并补充
python -m pipeline.fetch_kline --force
```

### 精确日期查询
```python
from pipeline.tushare_fetcher import TushareFetcher

fetcher = TushareFetcher(data_dir='data')

# 获取指定日期范围的精确数据
df = fetcher.fetch_stock_history(
    code='600519',
    start_date='2025-01-01',
    end_date='2025-12-31'
)

# 验证日期是否为交易日
if fetcher.is_trade_date('2025-07-15'):
    # 获取该交易日数据
    df = fetcher.fetch_for_date(['600519'], '2025-07-15')
```

## 与旧版本的差异

| 特性 | 旧版本 | 新版本（严格模式） |
|------|--------|-------------------|
| 缓存判断依据 | `update_cache.json` (今天是否更新) | 实际日期范围覆盖 |
| 交易日验证 | 无 | 使用 Tushare 交易日历 |
| 数据补充 | 增量获取最近 N 天 | 精确获取缺失日期范围 |
| API 失败降级 | 返回本地缓存 | 返回空字典（股票列表） |
| 数据优先级 | 新数据优先 | 现有数据优先（避免重复） |

## 注意事项

1. **Tushare 积分限制**: 基础用户 100 次/分钟，已内置限速
2. **交易日历有效期**: 每年自动扩展 ±1 年范围
3. **数据精确性**: 不使用估算、不猜测、不匹配错误日期
