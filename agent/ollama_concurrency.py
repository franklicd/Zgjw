"""
ollama_concurrency.py
~~~~~~~~~~~~~~~~~~~~~
Ollama 本地模型并发控制模块

Ollama 本地部署无 API 限流，但需考虑：
- GPU 显存限制（决定最大并发数）
- CPU/内存资源
- 磁盘 I/O（模型加载）
- 请求队列管理

建议配置：
- 8GB 显存：max_workers=2-4
- 12GB 显存：max_workers=4-8
- 16GB+ 显存：max_workers=8-16
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional


# ────────────────────────────────────────────────
# Ollama 本地模型推荐配置
# ────────────────────────────────────────────────

@dataclass
class OllamaConcurrencyConfig:
    """Ollama 本地模型并发配置"""
    # 最大并发 worker 数
    # 根据 GPU 显存调整：
    # - qwen3-vl:8b (Q4_K_M) 约需 6GB 显存
    # - 并发时会共享模型权重，但每个请求需要额外的推理显存
    max_concurrent_workers: int = 4

    # 请求间隔（秒），避免瞬间请求洪峰
    request_delay: float = 0.2

    # 重试配置
    max_retries: int = 3
    retry_base_delay: float = 1.0  # 指数退避基数

    # 进度追踪
    enable_progress_bar: bool = True
    progress_update_interval: int = 1  # 每 N 个请求更新一次进度

    # 描述（可选）
    description: str = ""


# 预设配置档位
CONSERVATIVE_CONFIG = OllamaConcurrencyConfig(
    max_concurrent_workers=2,
    request_delay=0.5,
    description="保守档：适合 8GB 显存或 CPU 推理"
)

STANDARD_CONFIG = OllamaConcurrencyConfig(
    max_concurrent_workers=4,
    request_delay=0.3,
    description="标准档：适合 12GB 显存"
)

AGGRESSIVE_CONFIG = OllamaConcurrencyConfig(
    max_concurrent_workers=8,
    request_delay=0.1,
    description="激进档：适合 16GB+ 显存"
)


class OllamaConcurrencyController:
    """
    Ollama 本地模型并发控制器

    功能：
    - 信号量控制最大并发数
    - 可选的请求延迟
    - 进度追踪和统计
    """

    def __init__(self, config: OllamaConcurrencyConfig = STANDARD_CONFIG):
        self.config = config
        self.semaphore = threading.Semaphore(config.max_concurrent_workers)
        self.lock = threading.Lock()

        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.pending_requests = 0

        # 时间追踪
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取并发许可（受信号量限制）

        Args:
            timeout: 超时时间（秒），None 表示无限等待

        Returns:
            True 表示获取成功，False 表示超时
        """
        if self.start_time is None:
            self.start_time = time.time()

        with self.lock:
            self.pending_requests += 1

        acquired = self.semaphore.acquire(blocking=True, timeout=timeout)

        if acquired:
            with self.lock:
                self.pending_requests -= 1
                self.total_requests += 1

        return acquired

    def release(self):
        """释放并发许可"""
        self.semaphore.release()

    def record_success(self):
        """记录成功请求"""
        with self.lock:
            self.successful_requests += 1

    def record_failure(self):
        """记录失败请求"""
        with self.lock:
            self.failed_requests += 1

    def get_stats(self) -> dict:
        """获取当前统计信息"""
        with self.lock:
            elapsed = time.time() - self.start_time if self.start_time else 0
            return {
                'total': self.total_requests,
                'success': self.successful_requests,
                'failed': self.failed_requests,
                'pending': self.pending_requests,
                'elapsed': elapsed,
                'qps': self.total_requests / elapsed if elapsed > 0 else 0,
            }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("Ollama 本地模型并发统计")
        print("=" * 50)
        print(f"最大并发数：{self.config.max_concurrent_workers}")
        print(f"请求间隔：{self.config.request_delay}秒")
        print(f"总请求数：{stats['total']}")
        print(f"成功：{stats['success']} | 失败：{stats['failed']}")
        print(f"总耗时：{stats['elapsed']:.1f}秒")
        print(f"QPS: {stats['qps']:.2f}")
        print("=" * 50)


# ────────────────────────────────────────────────
# 使用示例
# ────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试代码
    controller = OllamaConcurrencyController(STANDARD_CONFIG)

    def worker(task_id: int):
        if controller.acquire(timeout=30):
            try:
                print(f"[开始] 任务 {task_id}")
                time.sleep(1)  # 模拟推理
                controller.record_success()
                print(f"[完成] 任务 {task_id}")
            except Exception as e:
                controller.record_failure()
                print(f"[失败] 任务 {task_id}: {e}")
            finally:
                controller.release()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        concurrent.futures.wait(futures)

    controller.print_stats()
