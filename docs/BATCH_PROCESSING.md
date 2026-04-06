# 批量处理功能改进

## 功能特性

### 1. 分批处理
- 将股票分析请求按批次处理，每批最多10个请求
- 支持批量间延时，避免API限制

### 2. 断网重连支持
- 保存批量作业状态到本地文件
- 重启后自动检查挂起的作业
- 继续处理未完成的批量任务

## 使用方法

### 运行批量分析
```bash
# 正常运行
python agent/qwen_batch_review.py

# 指定批次大小
python agent/qwen_batch_review.py --batch-size 5

# 调整批次间延时
python agent/qwen_batch_review.py --batch-delay 30
```

### 检查挂起的作业
```bash
# 查看所有挂起的批量作业状态
python tools/check_batch_status.py

# 下载已完成作业的结果并清理状态文件
python tools/check_batch_status.py --download-results --cleanup
```

## 配置文件

可在 `config/qwen_review.yaml` 中配置以下参数：
```yaml
# 批量大小：每个批量处理多少个请求
batch_size: 10

# 批量延迟：批量处理的时间间隔（秒）
batch_delay: 60
```

## 作业状态管理

批量作业状态保存在 `data/batch_status/` 目录中，包含：
- 批量作业ID
- 请求数量
- 创建时间
- 当前状态

## 网络中断恢复

1. 当网络中断或程序终止时，已完成的批量作业状态会被保存
2. 重新运行脚本时，会自动检查并处理挂起的作业
3. 用户也可以使用 `check_batch_status.py` 工具手动管理作业状态