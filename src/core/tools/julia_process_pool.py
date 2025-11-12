# Copyright (c) Yogesh Singla and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Julia Process Pool with Worker Monitoring and Auto-Restart.

NEW FEATURES:
- Worker health monitoring with periodic checks
- Automatic detection and restart of hung workers
- Worker execution time tracking
- Forceful termination of stuck processes
- Comprehensive metrics and logging
"""

import atexit
import logging
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.env_server.types import CodeExecResult

logger = logging.getLogger(__name__)


@dataclass
class WorkerMetrics:
    """Metrics for worker health monitoring."""

    worker_id: int
    total_executions: int
    successful_executions: int
    failed_executions: int
    timeouts: int
    last_execution_start: float
    last_execution_end: float
    total_busy_time: float
    created_at: float
    restart_count: int


class JuliaWorkerProcess:
    """
    Single Julia worker process with health monitoring.

    IMPROVEMENTS:
    - Tracks execution start time to detect stuck workers
    - Records detailed metrics for monitoring
    - Provides health check interface
    """

    START_OUTPUT = "<<<START_OUTPUT>>>"
    START_ERROR = "<<<START_ERROR>>>"
    EXIT_CODE_PREFIX = "<<<EXIT_CODE:"
    END_EXECUTION = "<<<END_EXECUTION>>>"
    END_CODE = "<<<END_CODE>>>"

    # Worker health thresholds
    MAX_EXECUTION_TIME = 120  # 5 minutes - force kill after this

    def __init__(
        self,
        worker_id: int,
        julia_path: str,
        worker_script: str,
        optimization_flags: bool = True,
        recycle_after: int = 100,
        max_execution_time: int = 120,
    ):
        self.worker_id = worker_id
        self.julia_path = julia_path
        self.worker_script = worker_script
        self.optimization_flags = optimization_flags
        self.process: Optional[subprocess.Popen] = None
        self.is_busy = False
        self.is_healthy = True
        self.lock = threading.Lock()

        # Enhanced metrics tracking
        self.metrics = WorkerMetrics(
            worker_id=worker_id,
            total_executions=0,
            successful_executions=0,
            failed_executions=0,
            timeouts=0,
            last_execution_start=0.0,
            last_execution_end=time.time(),
            total_busy_time=0.0,
            created_at=time.time(),
            restart_count=0,
        )

        # Worker recycling and health
        self.recycle_after = recycle_after
        self.max_execution_time = max_execution_time

        self._start_process()

    def _start_process(self) -> None:
        """Start the Julia worker process."""
        cmd = [self.julia_path]
        julia_threads = os.environ.get("JULIA_NUM_THREADS", "2")

        if self.optimization_flags:
            cmd.extend(
                [
                    f"--threads={julia_threads}",
                    "--compile=min",
                    "--optimize=2",
                    "--startup-file=no",
                    "--history-file=no",
                ]
            )

        cmd.append(self.worker_script)

        try:
            env = os.environ.copy()
            env["JULIA_NUM_THREADS"] = julia_threads
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"

            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
            )

            ready_msg = self.process.stderr.readline()
            if "ready" not in ready_msg.lower():
                raise RuntimeError(
                    f"Worker {self.worker_id} did not start properly: {ready_msg}"
                )

            self.is_healthy = True
            logger.info(
                f"Worker {self.worker_id} started (PID: {self.process.pid}, "
                f"threads: {julia_threads}, restart_count: {self.metrics.restart_count})"
            )

        except Exception as e:
            self.is_healthy = False
            logger.error(f"Failed to start worker {self.worker_id}: {e}")
            raise

    def is_stuck(self) -> bool:
        """
        Check if worker is stuck (executing for too long).

        Returns:
            True if worker has been executing for longer than max_execution_time
        """
        if not self.is_busy:
            return False

        execution_duration = time.time() - self.metrics.last_execution_start
        is_stuck = execution_duration > self.max_execution_time

        if is_stuck:
            logger.warning(
                f"Worker {self.worker_id} is STUCK! "
                f"Executing for {execution_duration:.1f}s (limit: {self.max_execution_time}s)"
            )

        return is_stuck

    def force_kill(self) -> None:
        """
        Forcefully kill a stuck worker process.

        This is called by the health monitor when a worker is detected as stuck.
        """
        logger.error(
            f"FORCE KILLING worker {self.worker_id} "
            f"(stuck for {time.time() - self.metrics.last_execution_start:.1f}s)"
        )

        with self.lock:
            self.is_healthy = False
            self.metrics.timeouts += 1
            self.metrics.failed_executions += 1
            self._kill_process()

    def execute(self, code: str, timeout: int = 60) -> CodeExecResult:
        """Execute Julia code with enhanced monitoring."""
        with self.lock:
            if not self.is_healthy or self.process is None:
                raise RuntimeError(f"Worker {self.worker_id} is not healthy")

            # Check if worker needs recycling
            if (
                self.recycle_after > 0
                and self.metrics.total_executions >= self.recycle_after
            ):
                logger.info(
                    f"Worker {self.worker_id} reached {self.metrics.total_executions} executions "
                    f"(limit: {self.recycle_after}) - marking for recycle"
                )
                self.is_healthy = False
                raise RuntimeError(f"Worker {self.worker_id} needs recycling")

            self.is_busy = True
            self.metrics.total_executions += 1
            self.metrics.last_execution_start = time.time()

            result_container = {}
            timeout_occurred = threading.Event()

            def read_output():
                """Read output from worker process."""
                try:
                    self.process.stdin.write(code + "\n")
                    self.process.stdin.write(self.END_CODE + "\n")
                    self.process.stdin.flush()

                    stdout_lines = []
                    stderr_lines = []
                    exit_code = -1
                    current_section = None

                    while True:
                        if timeout_occurred.is_set():
                            logger.warning(
                                f"Worker {self.worker_id} read interrupted by timeout"
                            )
                            break

                        try:
                            line = self.process.stdout.readline()

                            if not line:
                                logger.error(
                                    f"Worker {self.worker_id} died unexpectedly"
                                )
                                result_container["error"] = (
                                    "Worker process died unexpectedly"
                                )
                                break

                            line = line.rstrip("\n")

                            if line == self.START_OUTPUT:
                                current_section = "stdout"
                                continue
                            elif line == self.START_ERROR:
                                current_section = "stderr"
                                continue
                            elif line.startswith(self.EXIT_CODE_PREFIX):
                                exit_code_str = line[len(self.EXIT_CODE_PREFIX) : -3]
                                exit_code = int(exit_code_str)
                                continue
                            elif line == self.END_EXECUTION:
                                break

                            if current_section == "stdout":
                                stdout_lines.append(line)
                            elif current_section == "stderr":
                                stderr_lines.append(line)

                        except Exception as e:
                            logger.error(
                                f"Error reading from worker {self.worker_id}: {e}"
                            )
                            result_container["error"] = (
                                f"Error reading from worker: {str(e)}"
                            )
                            break

                    stdout_str = "\n".join(stdout_lines) + (
                        "\n" if stdout_lines else ""
                    )
                    stderr_str = "\n".join(stderr_lines) + (
                        "\n" if stderr_lines else ""
                    )

                    result_container["result"] = CodeExecResult(
                        stdout=stdout_str,
                        stderr=stderr_str,
                        exit_code=exit_code,
                    )

                except Exception as e:
                    logger.error(f"Worker {self.worker_id} execution thread error: {e}")
                    result_container["error"] = f"Execution error: {str(e)}"

            try:
                reader_thread = threading.Thread(target=read_output, daemon=True)
                reader_thread.start()
                reader_thread.join(timeout=timeout)

                if reader_thread.is_alive():
                    # TIMEOUT
                    logger.error(
                        f"Worker {self.worker_id} TIMEOUT after {timeout}s "
                        f"(execution #{self.metrics.total_executions})"
                    )
                    timeout_occurred.set()
                    self.is_healthy = False
                    self.metrics.timeouts += 1
                    self.metrics.failed_executions += 1
                    self._kill_process()
                    reader_thread.join(timeout=1.0)

                    return CodeExecResult(
                        stdout="",
                        stderr=f"Worker timeout after {timeout}s - worker killed and will be recycled",
                        exit_code=-1,
                    )

                if "error" in result_container:
                    self.is_healthy = False
                    self.metrics.failed_executions += 1
                    return CodeExecResult(
                        stdout="",
                        stderr=result_container["error"],
                        exit_code=-1,
                    )

                if "result" in result_container:
                    self.metrics.successful_executions += 1
                    return result_container["result"]

                self.is_healthy = False
                self.metrics.failed_executions += 1
                return CodeExecResult(
                    stdout="",
                    stderr="Worker completed but returned no result",
                    exit_code=-1,
                )

            finally:
                self.metrics.last_execution_end = time.time()
                execution_time = (
                    self.metrics.last_execution_end - self.metrics.last_execution_start
                )
                self.metrics.total_busy_time += execution_time
                self.is_busy = False

    def _kill_process(self) -> None:
        """Kill the worker process."""
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except:
                try:
                    self.process.kill()
                    self.process.wait(timeout=1.0)
                except:
                    pass

    def shutdown(self) -> None:
        """Shutdown the worker process gracefully."""
        with self.lock:
            if self.process is not None:
                logger.info(f"Shutting down worker {self.worker_id}")
                self._kill_process()
                self.process = None
                self.is_healthy = False


class JuliaProcessPool:
    """
    Enhanced Julia Process Pool with Health Monitoring and Auto-Restart.

    NEW FEATURES:
    - Background health monitor thread
    - Automatic detection of stuck workers
    - Forceful restart of unhealthy workers
    - Request queue with backpressure
    - Comprehensive metrics
    """

    def __init__(
        self,
        size: int = 4,
        timeout: int = 60,
        julia_path: Optional[str] = None,
        optimization_flags: bool = True,
        auto_recover: bool = True,
        recycle_after: int = 100,
        max_execution_time: int = 120,
        health_check_interval: float = 10.0,
    ):
        """
        Initialize the Julia process pool with health monitoring.

        Args:
            size: Number of worker processes
            timeout: Default timeout for code execution (seconds)
            julia_path: Path to Julia executable
            optimization_flags: Enable Julia optimization flags
            auto_recover: Automatically restart failed workers
            recycle_after: Recycle workers after N executions
            max_execution_time: Maximum time a worker can execute before being killed (seconds)
            health_check_interval: How often to check worker health (seconds)
        """
        self.size = size
        self.timeout = timeout
        self.optimization_flags = optimization_flags
        self.auto_recover = auto_recover
        self.recycle_after = recycle_after
        self.max_execution_time = max_execution_time
        self.health_check_interval = health_check_interval

        if julia_path is None:
            julia_path = self._find_julia_executable()

        self.julia_path = julia_path
        self.worker_script = self._find_worker_script()

        # Worker management
        self.workers: list[JuliaWorkerProcess] = []
        self.available_workers: deque[JuliaWorkerProcess] = deque()
        self.pool_lock = threading.Lock()
        self.worker_available = threading.Condition(self.pool_lock)
        self.shutdown_flag = False

        # Health monitoring
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.stuck_workers_detected = 0
        self.workers_restarted = 0

        # Pool metrics
        self.total_requests = 0
        self.rejected_requests = 0
        self.queue_full_events = 0

        # Initialize workers
        logger.info(f"Creating Julia process pool with {size} workers")
        os.environ["JULIAUP_SKIP_VERSIONDB_UPDATE"] = "1"

        batch_size = 4
        for batch_start in range(0, size, batch_size):
            batch_end = min(batch_start + batch_size, size)
            batch_workers = []

            def create_worker(worker_id):
                try:
                    worker = JuliaWorkerProcess(
                        worker_id=worker_id,
                        julia_path=self.julia_path,
                        worker_script=self.worker_script,
                        optimization_flags=self.optimization_flags,
                        recycle_after=self.recycle_after,
                        max_execution_time=self.max_execution_time,
                    )
                    return (worker_id, worker, None)
                except Exception as e:
                    return (worker_id, None, e)

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=batch_size
            ) as executor:
                futures = {
                    executor.submit(create_worker, i): i
                    for i in range(batch_start, batch_end)
                }

                for future in concurrent.futures.as_completed(futures):
                    worker_id, worker, error = future.result()

                    if error:
                        logger.error(f"Failed to create worker {worker_id}: {error}")
                        self.shutdown()
                        raise RuntimeError(f"Failed to create worker pool: {error}")

                    batch_workers.append((worker_id, worker))

            batch_workers.sort(key=lambda x: x[0])
            for worker_id, worker in batch_workers:
                self.workers.append(worker)
                self.available_workers.append(worker)
                logger.debug(f"Worker {worker_id} added to pool")

            if batch_end < size:
                time.sleep(0.1)

        logger.info(f"Julia process pool initialized with {len(self.workers)} workers")

        # Start health monitor
        self._start_health_monitor()

        atexit.register(self.shutdown)

    def _start_health_monitor(self) -> None:
        """Start background thread for worker health monitoring."""

        def health_monitor():
            """Background health monitor that checks for stuck workers."""
            logger.info(
                f"Health monitor started (check_interval={self.health_check_interval}s, "
                f"max_execution_time={self.max_execution_time}s)"
            )

            while not self.shutdown_flag:
                try:
                    time.sleep(self.health_check_interval)

                    if self.shutdown_flag:
                        break

                    with self.pool_lock:
                        for worker in self.workers:
                            # Check if worker is stuck
                            if worker.is_stuck():
                                self.stuck_workers_detected += 1
                                logger.error(
                                    f"🚨 STUCK WORKER DETECTED: Worker {worker.worker_id} "
                                    f"has been executing for "
                                    f"{time.time() - worker.metrics.last_execution_start:.1f}s"
                                )

                                # Force kill the stuck worker
                                worker.force_kill()

                                # Try to restart if auto_recover is enabled
                                if self.auto_recover and not self.shutdown_flag:
                                    try:
                                        logger.info(
                                            f"Attempting to restart stuck worker {worker.worker_id}"
                                        )
                                        worker.shutdown()

                                        new_worker = JuliaWorkerProcess(
                                            worker_id=worker.worker_id,
                                            julia_path=self.julia_path,
                                            worker_script=self.worker_script,
                                            optimization_flags=self.optimization_flags,
                                            recycle_after=self.recycle_after,
                                            max_execution_time=self.max_execution_time,
                                        )
                                        new_worker.metrics.restart_count = (
                                            worker.metrics.restart_count + 1
                                        )

                                        # Replace worker in pool
                                        self.workers[worker.worker_id] = new_worker
                                        self.available_workers.append(new_worker)
                                        self.workers_restarted += 1

                                        logger.info(
                                            f"✅ Successfully restarted worker {worker.worker_id} "
                                            f"(restart #{new_worker.metrics.restart_count})"
                                        )

                                        # Notify waiting threads
                                        self.worker_available.notify()

                                    except Exception as e:
                                        logger.error(
                                            f"Failed to restart worker {worker.worker_id}: {e}"
                                        )

                except Exception as e:
                    if not self.shutdown_flag:
                        logger.error(f"Health monitor error: {e}")

            logger.info("Health monitor stopped")

        self.health_monitor_thread = threading.Thread(
            target=health_monitor, daemon=True, name="julia_health_monitor"
        )
        self.health_monitor_thread.start()
        logger.info("Health monitor thread started")

    def _find_julia_executable(self) -> str:
        """Find Julia executable."""
        julia_bin_dirs = [
            os.path.expanduser("~/.juliaup/bin"),
            os.path.expanduser("~/.julia/bin"),
            "/usr/local/bin",
            "/usr/bin",
        ]

        current_path = os.environ.get("PATH", "")
        path_entries = current_path.split(os.pathsep)

        for julia_path in julia_bin_dirs:
            if os.path.isdir(julia_path) and julia_path not in path_entries:
                current_path = f"{julia_path}{os.pathsep}{current_path}"

        os.environ["PATH"] = current_path

        julia_path = shutil.which("julia")
        if julia_path:
            return julia_path

        common_paths = [
            os.path.expanduser("~/.juliaup/bin/julia"),
            os.path.expanduser("~/.julia/bin/julia"),
            "/usr/local/bin/julia",
            "/usr/bin/julia",
        ]

        for path in common_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        raise RuntimeError(
            "Julia executable not found. Please install Julia: "
            "https://julialang.org/downloads/"
        )

    def _find_worker_script(self) -> str:
        """Find the julia_repl_worker.jl script."""
        this_dir = Path(__file__).parent
        worker_script = this_dir / "julia_repl_worker.jl"

        if worker_script.exists():
            return str(worker_script)

        raise RuntimeError(
            f"Worker script not found at {worker_script}. "
            "Please ensure julia_repl_worker.jl is in the same directory."
        )

    def _get_available_worker(
        self, timeout: float = 60.0
    ) -> Optional[JuliaWorkerProcess]:
        """Get an available worker with timeout."""
        deadline = time.time() + timeout

        with self.worker_available:
            while True:
                while self.available_workers:
                    worker = self.available_workers.popleft()

                    if worker.is_healthy:
                        logger.debug(
                            f"Allocated worker {worker.worker_id} "
                            f"({len(self.available_workers)} remaining)"
                        )
                        return worker

                    if self.auto_recover and not self.shutdown_flag:
                        logger.warning(
                            f"Worker {worker.worker_id} is unhealthy, attempting recovery"
                        )
                        try:
                            worker.shutdown()
                            worker = JuliaWorkerProcess(
                                worker_id=worker.worker_id,
                                julia_path=self.julia_path,
                                worker_script=self.worker_script,
                                optimization_flags=self.optimization_flags,
                                recycle_after=self.recycle_after,
                                max_execution_time=self.max_execution_time,
                            )
                            worker.metrics.restart_count += 1
                            self.workers[worker.worker_id] = worker
                            self.workers_restarted += 1
                            logger.info(
                                f"Worker {worker.worker_id} recovered successfully"
                            )
                            return worker
                        except Exception as e:
                            logger.error(
                                f"Failed to recover worker {worker.worker_id}: {e}"
                            )

                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    logger.error(
                        f"Timeout waiting for available worker after {timeout}s "
                        f"(pool size: {self.size}, all workers busy)"
                    )
                    return None

                logger.debug(
                    f"All {self.size} workers busy, waiting up to {remaining_time:.1f}s..."
                )
                if not self.worker_available.wait(timeout=remaining_time):
                    logger.error(
                        f"Timeout waiting for available worker after {timeout}s"
                    )
                    return None

    def _return_worker(self, worker: JuliaWorkerProcess) -> None:
        """Return a worker to the pool."""
        with self.worker_available:
            if worker.is_healthy and not self.shutdown_flag:
                self.available_workers.append(worker)
                logger.debug(
                    f"Returned worker {worker.worker_id} "
                    f"({len(self.available_workers)} available)"
                )
                self.worker_available.notify()

    def execute(self, code: str, timeout: Optional[int] = None) -> CodeExecResult:
        """Execute Julia code using pool."""
        if self.shutdown_flag:
            return CodeExecResult(
                stdout="",
                stderr="Process pool has been shut down",
                exit_code=-1,
            )

        if timeout is None:
            timeout = self.timeout

        self.total_requests += 1

        worker_wait_timeout = min(60.0, timeout)
        worker = self._get_available_worker(timeout=worker_wait_timeout)

        if worker is None:
            self.rejected_requests += 1
            logger.error(
                f"Failed to acquire worker within {worker_wait_timeout}s "
                f"(pool size: {self.size}, all workers busy)"
            )
            return CodeExecResult(
                stdout="",
                stderr=f"No available worker (timeout waiting for worker after {worker_wait_timeout}s). "
                f"All {self.size} workers are busy. Consider increasing pool size or reducing execution time.",
                exit_code=-1,
            )

        try:
            logger.debug(f"Executing code in worker {worker.worker_id}")
            result = worker.execute(code, timeout=timeout)
            logger.debug(
                f"Worker {worker.worker_id} completed (exit_code={result.exit_code})"
            )
            return result

        finally:
            self._return_worker(worker)

    def get_metrics(self) -> dict:
        """Get pool and worker metrics."""
        with self.pool_lock:
            worker_metrics = []
            for worker in self.workers:
                worker_metrics.append(
                    {
                        "worker_id": worker.worker_id,
                        "is_busy": worker.is_busy,
                        "is_healthy": worker.is_healthy,
                        "total_executions": worker.metrics.total_executions,
                        "successful_executions": worker.metrics.successful_executions,
                        "failed_executions": worker.metrics.failed_executions,
                        "timeouts": worker.metrics.timeouts,
                        "restart_count": worker.metrics.restart_count,
                        "total_busy_time": worker.metrics.total_busy_time,
                        "avg_execution_time": (
                            worker.metrics.total_busy_time
                            / worker.metrics.total_executions
                            if worker.metrics.total_executions > 0
                            else 0
                        ),
                    }
                )

            return {
                "pool_size": self.size,
                "available_workers": len(self.available_workers),
                "busy_workers": self.size - len(self.available_workers),
                "total_requests": self.total_requests,
                "rejected_requests": self.rejected_requests,
                "stuck_workers_detected": self.stuck_workers_detected,
                "workers_restarted": self.workers_restarted,
                "workers": worker_metrics,
            }

    def shutdown(self) -> None:
        """Shutdown pool and health monitor."""
        if self.shutdown_flag:
            return

        logger.info("Shutting down Julia process pool")
        self.shutdown_flag = True

        # Wait for health monitor to stop
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            logger.info("Waiting for health monitor to stop...")
            self.health_monitor_thread.join(timeout=5.0)

        with self.pool_lock:
            for worker in self.workers:
                worker.shutdown()

            self.workers.clear()
            self.available_workers.clear()

        logger.info("Julia process pool shutdown complete")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __del__(self):
        self.shutdown()
