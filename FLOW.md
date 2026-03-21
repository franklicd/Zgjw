# AgentTrader 代码运行流程详解

本文档说明 `run_all.py` 一键脚本背后每个步骤的职责、关键文件，以及图形对比分析发生的位置。

---

## 总览

```
步骤 1  pipeline/fetch_kline.py          拉取 K 线原始数据
    ↓
步骤 2  pipeline/cli.py preselect        量化规则初选
    ↓
步骤 3  dashboard/export_kline_charts.py  生成候选股 K 线图片
    ↓
步骤 4  agent/doubao_review.py             AI 视觉分析打分        ← 图形对比核心
    ↓
步骤 5  打印推荐结果
```

---

## 步骤 1：拉取 K 线数据

**文件：** `pipeline/fetch_kline.py`
**配置：** `config/fetch_kline.yaml`
**输出：** `data/raw/<code>.csv`

从 `pipeline/stocklist.csv` 读取股票列表，多线程并发调用 **AKShare** 接口，把每只股票的日线前复权数据（日期、开、收、高、低、成交量）保存为单独的 CSV 文件。

关键配置：

| 参数 | 说明 |
|---|---|
| `start` / `end` | 抓取日期范围（YYYYMMDD 或 `today`）|
| `stocklist` | 股票池 CSV 路径 |
| `exclude_boards` | 排除板块（`gem` 创业板 / `star` 科创板 / `bj` 北交所）|
| `workers` | 并发线程数 |

---

## 步骤 2：量化规则初选

**文件：** `pipeline/cli.py` → `pipeline/select_stock.py` → `pipeline/Selector.py`
**配置：** `config/rules_preselect.yaml`
**输出：** `data/candidates/candidates_latest.json`

读取 `data/raw/*.csv`，对每只股票运行两套量化策略，筛出当日满足条件的候选股。

### B1 策略

同时满足以下全部条件：

1. **KDJ 低位**：今日 J 值 < 阈值，或处于历史低分位（expanding 分位数，无未来泄漏）
2. **知行均线**：短期均线（zxdq）> 长期均线（zxdkx），且收盘价 > 长期均线
3. **周线多头排列**：周线 MA_short > MA_mid > MA_long
4. **最大量非阴线**：过去 N 日内成交量最大那天不是阴线

### 砖型图策略

同时满足以下全部条件：

1. **砖型图形态**：今日红柱，昨日绿柱，且红柱高度 ≥ 昨日绿柱 × 增长比例，前方有连续绿柱
2. **当日涨幅未超限**：收盘涨幅 < 阈值（防追高）
3. **知行线位置**：收盘价 < zxdq × 比例（未过热）
4. **zxdq > zxdkx**：短均线在长均线上方
5. **周线多头排列**（可配置关闭）

### 流动性过滤

两套策略均只在「流动性池」内运行：按滚动 N 日成交额排名，取前 `top_m` 只，避免低流动性个股入选。

---

## 步骤 3：生成 K 线图片

**文件：** `dashboard/export_kline_charts.py` + `dashboard/components/charts.py`
**输出：** `data/kline/<pick_date>/<code>_day.jpg`

读取候选股的 CSV 日线数据，用 **Plotly** 绘制带均线和成交量柱的 K 线图，再通过 **kaleido** 引擎导出为 JPG 图片（1400×700，2× 分辨率）。

这一步是图形对比的**起点**：将数字数据转换成人眼（以及多模态模型）可以直接阅读的图像。

---

## 步骤 4：AI 视觉分析打分（图形对比核心）

**文件：** `agent/doubao_review.py`
**提示词：** `agent/prompt.md`
**配置：** `config/doubao_review.yaml`
**输出：** `data/review/<pick_date>/<code>.json` + `suggestion.json`

### 为什么要看图

量化规则（步骤 2）只能计算数值信号，无法判断：

- 这次突破是真突破还是假突破？
- 放量是主力建仓还是出货？
- 当前位置是低位还是历史高位？

这些问题需要结合**整体图形形态**做主观判断，因此引入多模态大模型直接「看图」。

### 打分流程

```
K 线 JPG 图片（base64 编码）
    + prompt.md（交易员评分框架）
    ↓
Qwen3.5-plus（视觉语言模型）
    ↓
强制推理（必须先完成再打分）：
  trend_reasoning        趋势结构推理
  position_reasoning     价格位置推理
  volume_reasoning       量价行为推理
  abnormal_move_reasoning 前期异动推理
  signal_reasoning       信号判断推理
    ↓
JSON 评分结果
```

### 四个评分维度

| 维度 | 权重 | 评估内容 |
|---|---|---|
| 趋势结构 `trend_structure` | 20% | 均线多空排列、斜率、价格与均线关系 |
| 价格位置 `price_position` | 20% | 是否在低位 / 中位 / 压力区 / 历史高位 |
| 量价行为 `volume_behavior` | 30% | 上涨放量、回调缩量、有无放量大阴线 |
| 前期异动 `previous_abnormal_move` | 30% | 有无主力建仓大阳线、突破结构、涨幅是否过大 |

每个维度 1～5 分，加权求和得 `total_score`。

### 判定规则

| 判定 | 条件 |
|---|---|
| **PASS**（推荐） | total_score ≥ 4.0 |
| **WATCH**（观察） | 3.2 ≤ total_score < 4.0 |
| **FAIL**（淘汰） | total_score < 3.2，或 volume_behavior = 1 |

### 信号类型

- `trend_start`：主升启动
- `rebound`：跌后反弹
- `distribution_risk`：出货风险

---

## 步骤 5：输出推荐结果

读取 `data/review/<pick_date>/suggestion.json`，按分数降序打印达到门槛的推荐股票，包含排名、代码、总分、信号类型、研判和交易员点评。

---

## 数据流转示意

```
pipeline/stocklist.csv
        ↓
data/raw/<code>.csv          ← 步骤1 AKShare 日线数据
        ↓
data/candidates/candidates_latest.json  ← 步骤2 量化初选候选
        ↓
data/kline/<date>/<code>_day.jpg        ← 步骤3 K线图片
        ↓
data/review/<date>/<code>.json          ← 步骤4 AI单股评分
data/review/<date>/suggestion.json      ← 步骤4 汇总推荐
```

---

## 关键模块索引

| 模块 | 路径 | 职责 |
|---|---|---|
| K线下载 | `pipeline/fetch_kline.py` | AKShare 日线抓取，多线程 |
| 量化选股 | `pipeline/Selector.py` | B1 / 砖型图信号计算 |
| 数据预处理 | `pipeline/pipeline_core.py` | 多进程特征计算、流动性池 |
| 初选入口 | `pipeline/cli.py` | 命令行接口，写入候选文件 |
| 图表绘制 | `dashboard/components/charts.py` | Plotly K线图生成 |
| 图表导出 | `dashboard/export_kline_charts.py` | JPG 批量导出 |
| AI 评审基类 | `agent/base_reviewer.py` | 候选加载、结果写入、汇总逻辑 |
| Qwen 评审 | `agent/doubao_review.py` | 调用 Doubao 视觉模型打分 |
| 评分提示词 | `agent/prompt.md` | 交易员评分框架（四维度 + 输出格式）|
| 全流程入口 | `run_all.py` | 串联步骤 1-5 |
