"""
qwen_concurrency.py
~~~~~~~~~~~~~~~~~~~
阿里云通义千问（Qwen）API 并发控制模块

根据阿里云官方限流标准设计：
https://help.aliyun.com/zh/model-studio/developer-reference/rate-limit

一、实时 API（中国内地）
- RPM（每分钟请求数）：30,000
- TPM（每分钟总 Token，输入 + 输出）：5,000,000
- RPS（每秒请求数，理论上限）：500
- TPS（每秒 Token，理论上限）：83,333

二、批量 API（Batch Chat）
- 排队请求上限：单账号单模型最多 10,000 个等待中的请求
- 提交频率上限：1,000 QPS（10 秒内最多 10,000 次提交）

注意：实际使用中应考虑图片 token 消耗（单张 K 线图约 1000-2000 tokens）
"""

import asyncio
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore


# ────────────────────────────────────────────────
# 限流配置常量
# ────────────────────────────────────────────────

@dataclass
class QwenRateLimitConfig:
    """Qwen API 限流配置"""
    # 实时 API 限流（中国内地）
    rpm_limit: int = 30000      # 每分钟请求数
    tpm_limit: int = 5000000    # 每分钟 token 数
    rps_limit: int = 500        # 每秒请求数
    tps_limit: int = 83333      # 每秒 token 数

    # 批量 API 限流
    batch_queue_limit: int = 10000  # 排队请求上限
    batch_qps_limit: int = 1000     # 提交频率 QPS

    # 安全余量（建议使用限流的 80%，避免触及上限）
    safety_factor: float = 0.8

    # 估算的每次请求 token 消耗（图片 + 文本）
    estimated_tokens_per_request: int = 2000

    @property
    def safe_rpm(self) -> int:
        return int(self.rpm_limit * self.safety_factor)

    @property
    def safe_tpm(self) -> int:
        return int(self.tpm_limit * self.safety_factor)

    @property
    def safe_rps(self) -> int:
        return int(self.rps_limit * self.safety_factor)

    @property
    def safe_tps(self) -> int:
        return int(self.tps_limit * self.safety_factor)

    @property
    def max_concurrent_workers(self) -> int:
        """
        根据限流计算最大并发 worker 数
        基于 RPS 限制和安全余量
        考虑每次请求约 2000 tokens，TPM 限制下每分钟最多 2500 次请求
        取 RPS 和 TPM 的较小值
        """
        # 基于 RPS 的并发数
        rps_based_workers = self.safe_rps

        # 基于 TPM 的并发数（考虑每次请求的 token 消耗）
        tpm_based_workers = self.safe_tpm // self.estimated_tokens_per_request // 60

        # 取较小值，但至少为 1
        return max(1, min(rps_based_workers, tpm_based_workers))


# ────────────────────────────────────────────────
# Token 桶限流器
# ────────────────────────────────────────────────

class TokenBucket:
    """
    Token 桶限流器实现

    用于平滑控制请求速率，支持突发流量
    """
    def __init__(self, rate: float, capacity: int):
        """
        Args:
            rate: 每秒补充的 token 数
            capacity: 桶容量（最大 token 数）
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """补充 token"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        获取 token

        Args:
            tokens: 需要的 token 数量
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            是否成功获取
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            # 短暂休眠，避免 CPU 空转
            time.sleep(0.01)

    async def acquire_async(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """异步版本"""
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._refill()
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return True

            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    return False

            await asyncio.sleep(0.01)


# ────────────────────────────────────────────────
# 滑动窗口限流器
# ────────────────────────────────────────────────

class SlidingWindowRateLimiter:
    """
    滑动窗口限流器

    用于精确控制 RPM/TPM 等分钟级别的限流
    """
    def __init__(self, limit: int, window_seconds: int = 60):
        """
        Args:
            limit: 窗口内的最大请求数
            window_seconds: 窗口大小（秒）
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = threading.Lock()

    def _cleanup(self):
        """清理过期请求"""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取许可

        Args:
            timeout: 超时时间（秒）

        Returns:
            是否成功获取
        """
        start_time = time.monotonic()

        while True:
            with self._lock:
                self._cleanup()
                if len(self.requests) < self.limit:
                    self.requests.append(time.monotonic())
                    return True

            if timeout is not None:
                if time.monotonic() - start_time >= timeout:
                    return False

            time.sleep(0.01)

    @property
    def current_count(self) -> int:
        """当前窗口内的请求数"""
        with self._lock:
            self._cleanup()
            return len(self.requests)

    @property
    def remaining(self) -> int:
        """剩余可用请求数"""
        return self.limit - self.current_count


# ────────────────────────────────────────────────
# Qwen API 并发控制器
# ────────────────────────────────────────────────

@dataclass
class ConcurrencyStats:
    """并发统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_time

    @property
    def requests_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.total_requests / elapsed if elapsed > 0 else 0

    @property
    def success_rate(self) -> float:
        return self.successful_requests / self.total_requests * 100 if self.total_requests > 0 else 0


class QwenConcurrencyController:
    """
    Qwen API 并发控制器

    综合使用多种限流策略：
    1. Token 桶：控制 RPS/TPS
    2. 滑动窗口：控制 RPM/TPM
    3. 信号量：控制最大并发数
    """
    def __init__(self, config: Optional[QwenRateLimitConfig] = None):
        """
        Args:
            config: 限流配置，None 使用默认配置
        """
        self.config = config or QwenRateLimitConfig()

        # RPS 限流器（每秒请求数）
        self.rps_bucket = TokenBucket(
            rate=self.config.safe_rps,
            capacity=self.config.safe_rps * 2  # 允许短暂突发
        )

        # RPM 限流器（每分钟请求数）
        self.rpm_limiter = SlidingWindowRateLimiter(
            limit=self.config.safe_rpm,
            window_seconds=60
        )

        # TPM 限流器（基于估算的 token 消耗）
        self.tpm_limiter = SlidingWindowRateLimiter(
            limit=self.config.safe_tpm,
            window_seconds=60
        )

        # 并发信号量
        self.semaphore = Semaphore(self.config.max_concurrent_workers)

        # 统计信息
        self.stats = ConcurrencyStats()
        self._stats_lock = threading.Lock()

        # 运行状态
        self._running = True

    def acquire(self, estimated_tokens: int = 2000, timeout: Optional[float] = None) -> bool:
        """
        获取 API 调用许可

        Args:
            estimated_tokens: 预估的 token 消耗
            timeout: 超时时间

        Returns:
            是否成功获取许可
        """
        # 获取并发信号量
        acquired = self.semaphore.acquire(timeout=timeout or 300)
        if not acquired:
            return False

        try:
            # RPS 限流
            if not self.rps_bucket.acquire(tokens=1, timeout=timeout):
                self.semaphore.release()
                return False

            # RPM 限流
            if not self.rpm_limiter.acquire(timeout=timeout):
                self.semaphore.release()
                return False

            # TPM 限流（基于估算）
            if not self.tpm_limiter.acquire(timeout=timeout):
                self.semaphore.release()
                return False

            # 更新统计
            with self._stats_lock:
                self.stats.total_requests += 1
                self.stats.total_tokens += estimated_tokens

            return True

        except Exception:
            self.semaphore.release()
            raise

    def release(self):
        """释放并发许可"""
        self.semaphore.release()

    def record_success(self, actual_tokens: int = 0):
        """记录成功请求"""
        with self._stats_lock:
            self.stats.successful_requests += 1
            if actual_tokens > 0:
                self.stats.total_tokens += actual_tokens

    def record_failure(self):
        """记录失败请求"""
        with self._stats_lock:
            self.stats.failed_requests += 1

    def get_stats(self) -> ConcurrencyStats:
        """获取统计信息"""
        with self._stats_lock:
            return ConcurrencyStats(
                total_requests=self.stats.total_requests,
                successful_requests=self.stats.successful_requests,
                failed_requests=self.stats.failed_requests,
                total_tokens=self.stats.total_tokens,
                start_time=self.stats.start_time
            )

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print(f"""
╔═══════════════════════════════════════════╗
║         Qwen API 并发统计                 ║
╠═══════════════════════════════════════════╣
║ 总请求数：    {stats.total_requests:>8}                    ║
║ 成功：        {stats.successful_requests:>8}                    ║
║ 失败：        {stats.failed_requests:>8}                    ║
║ 成功率：      {stats.success_rate:>7.2f}%                  ║
║ 总 Token 数：  {stats.total_tokens:>12}                ║
║ 运行时长：    {stats.elapsed_seconds:>8.1f} 秒               ║
║ 请求/秒：     {stats.requests_per_second:>7.2f}                    ║
║ 并发数上限：  {self.config.max_concurrent_workers:>8}                    ║
║ RPM 余量：   {self.rpm_limiter.remaining:>8}                    ║
║ TPM 余量：   {self.tpm_limiter.remaining:>8}                    ║
╚═══════════════════════════════════════════╝
""")


# ────────────────────────────────────────────────
# 装饰器版本（方便集成到现有代码）
# ────────────────────────────────────────────────

def with_qwen_rate_limit(controller: Optional[QwenConcurrencyController] = None,
                         estimated_tokens: int = 2000):
    """
    Qwen API 限流装饰器

    用法:
        controller = QwenConcurrencyController()

        @with_qwen_rate_limit(controller, estimated_tokens=2000)
        def call_qwen_api(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            if controller is None:
                # 没有限流器，直接调用
                return func(*args, **kwargs)

            # 获取许可
            if not controller.acquire(estimated_tokens=estimated_tokens, timeout=300):
                raise TimeoutError("获取 API 许可超时，请稍后重试")

            try:
                result = func(*args, **kwargs)
                controller.record_success()
                return result
            except Exception as e:
                controller.record_failure()
                raise
            finally:
                controller.release()

        return wrapper
    return decorator


# ────────────────────────────────────────────────
# 异步版本
# ────────────────────────────────────────────────

class AsyncQwenConcurrencyController:
    """异步版本的并发控制器"""

    def __init__(self, config: Optional[QwenRateLimitConfig] = None):
        self.config = config or QwenRateLimitConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_workers)
        self._stats = ConcurrencyStats()
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int = 2000):
        await self.semaphore.acquire()
        async with self._lock:
            self._stats.total_requests += 1
            self._stats.total_tokens += estimated_tokens

    def release(self):
        self.semaphore.release()

    async def record_success(self, actual_tokens: int = 0):
        async with self._lock:
            self._stats.successful_requests += 1
            if actual_tokens > 0:
                self._stats.total_tokens += actual_tokens

    async def record_failure(self):
        async with self._lock:
            self._stats.failed_requests += 1


# ────────────────────────────────────────────────
# 预设配置
# ────────────────────────────────────────────────

# 保守配置：适合小账号或测试环境
CONSERVATIVE_CONFIG = QwenRateLimitConfig(
    safety_factor=0.5,  # 使用 50% 的限流额度
    estimated_tokens_per_request=3000,
)

# 标准配置：适合生产环境
STANDARD_CONFIG = QwenRateLimitConfig(
    safety_factor=0.8,  # 使用 80% 的限流额度
    estimated_tokens_per_request=2500,
)

# 激进配置：适合大批量处理（需谨慎使用）
AGGRESSIVE_CONFIG = QwenRateLimitConfig(
    safety_factor=0.9,  # 使用 90% 的限流额度
    estimated_tokens_per_request=2000,
)
