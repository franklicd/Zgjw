# 豆包批量处理功能

## 功能特性

### 1. 分批处理
- 将股票分析请求按批次处理，每批最多10个请求
- 支持批量间延时，避免API限制

### 2. 简化设计
- 实时处理，无需复杂的续传功能
- 每个请求实时调用API
- 简单的状态跟踪

## 使用方法

### 运行批量分析
```bash
# 正常运行
python agent/doubao_batch_review.py

# 指定批次大小
python agent/doubao_batch_review.py --batch-size 5

# 调整批次间延时
python agent/doubao_batch_review.py --batch-delay 30
```

## 配置文件

可在 `config/doubao_review.yaml` 中配置以下参数：
```yaml
# 原有配置
model: doubao-seed-2.0-pro
request_delay: 3
skip_existing: false
suggest_min_score: 4.0

# 批量处理新增配置
batch_size: 10        # 每批处理的请求数量
batch_delay: 60       # 批次间的延迟（秒）
```

## 设计理念

本实现采用简化的分批处理方式：
- 按批次顺序处理请求
- 在批次间添加延迟
- 每个请求实时调用API

这种方式避免了复杂的批量API依赖，更适合豆包API的实时调用特性。