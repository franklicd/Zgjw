#!/bin/bash
set -euo pipefail

# 脚本配置
WORK_DIR="/Users/ghostwhisper/claudeWorkspace/StockTradebyZ"
SCRIPT_PATH="${WORK_DIR}/run_all.py"
LOG_FILE="${WORK_DIR}/run_log_$(date +%Y%m%d).log"
PYTHON_PATH="/usr/local/bin/python3"
# 记录日志函数
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 发送飞书通知函数（用OpenClaw自带飞书通道）
send_feishu_alert() {
    local message="$1"
    log "发送飞书告警：$message"
    
    /usr/local/bin/openclaw message send \
        --channel feishu \
        --message "【交易脚本告警】
时间：$(date +'%Y-%m-%d %H:%M:%S')
内容：$message" >> "$LOG_FILE" 2>&1
}

# 检查是否为中国法定节假日
is_holiday() {
    local year=$(date +%Y)
    local month=$(date +%m)
    local day=$(date +%d)
    
    # 方法1: 使用中国节假日API（免费接口）
    local holiday_status=$(curl -s "http://timor.tech/api/holiday/info/${year}-${month}-${day}" | jq -r '.type.type')
    
    # 0=工作日 1=节假日 2=调休补班
    if [ "$holiday_status" = "1" ]; then
        return 0 # 是节假日
    else
        return 1 # 不是节假日
    fi
}

# 检查是否为周一到周五
is_weekday() {
    local weekday=$(date +%w)
    # 1-5 代表周一到周五，0是周日，6是周六
    if [ "$weekday" -ge 1 ] && [ "$weekday" -le 5 ]; then
        return 0 # 是工作日
    else
        return 1 # 是周末
    fi
}

# 主逻辑
main() {
    cd "$WORK_DIR" || exit 1
    
    log "开始执行定时任务"
    
    # 检查是否需要执行
    if is_holiday; then
        log "今天是中国法定节假日，跳过执行"
        exit 0
    fi
    
    if ! is_weekday; then
        log "今天是周末，跳过执行"
        exit 0
    fi
    
    # 执行Python脚本
    log "开始执行run_all.py脚本"
    set +e
    "$PYTHON_PATH" "$SCRIPT_PATH" >> "$LOG_FILE" 2>&1
    local exit_code=$?
    set -e
    
    # 检查执行结果
    if [ $exit_code -eq 0 ]; then
        log "脚本执行成功"
    else
        log "脚本执行失败，退出码：$exit_code"
        send_feishu_alert "交易脚本执行失败！退出码：$exit_code\n日志路径：$LOG_FILE"
        exit 1
    fi
    
    log "任务执行完成"
}

main "$@"
