# 批量处理功能说明

## 功能特性

### 1. 分批处理
- 将股票分析请求按批次处理，每批最多10个请求（可配置）
- 支持批量间延时（可选）
- 支持多线程并发处理

### 2. 多模型支持
- **Qwen**: `agent/qwen_batch_review.py`
- **Doubao**: `agent/doubao_batch_review.py`

## 使用方法

### Qwen 批量分析
```bash
python agent/qwen_batch_review.py
python agent/qwen_batch_review.py --batch-size 5
python agent/qwen_batch_review.py --batch-delay 30
```

### Doubao 批量分析
```bash
# 设置 API Key
export DOUBAO_API_KEY="your-api-key"

python agent/doubao_batch_review.py
python agent/doubao_batch_review.py --batch-size 5
```

### 检查挂起的作业
```bash
python tools/check_batch_status.py
```

## 配置文件

### Qwen (config/qwen_review.yaml)
```yaml
batch_mode: true
batch_size: 10
batch_delay: 60
max_workers: 1
use_rate_limit: true
```

### Doubao (config/doubao_review.yaml)
```yaml
batch_size: 10
batch_delay: 0  # 0 表示不等待
```

## 注意事项

- Doubao 批量处理默认 `batch_delay: 0`（不等待）
- Qwen 批量处理默认 `batch_delay: 60`（配合限流保护）
- 两个模型都支持多线程并发