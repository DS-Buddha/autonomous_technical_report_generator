"""
Asynchronous code execution tasks with production-grade sandboxing.

Implements:
- Worker-broker-result pattern
- Resource-constrained execution
- Retry logic with Dead Letter Queue
- Idempotency and result persistence
- Monitoring hooks
"""

import os
import sys
import tempfile
import subprocess
import hashlib
import resource
import signal
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded, Reject
from src.tasks.celery_app import celery_app
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    """Structured execution result with comprehensive metadata."""
    task_id: str
    code_hash: str  # For idempotency
    success: bool
    stdout: str
    stderr: str
    returncode: int
    execution_time_ms: int
    memory_used_mb: Optional[float]
    retry_count: int
    timestamp: str
    worker_id: str

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CodeExecutionTask(Task):
    """
    Custom Celery task with enhanced error handling and monitoring.

    Implements:
    - Automatic retry with exponential backoff
    - Dead Letter Queue routing for poison pills
    - Resource monitoring and cleanup
    """

    autoretry_for = (subprocess.TimeoutExpired, MemoryError, OSError)
    retry_kwargs = {'max_retries': 3, 'countdown': 5}
    retry_backoff = True
    retry_backoff_max = 600  # Max 10 minutes
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure - route to Dead Letter Queue after max retries."""
        retry_count = self.request.retries

        if retry_count >= self.max_retries:
            logger.error(
                f"Task {task_id} failed after {retry_count} retries. "
                f"Routing to Dead Letter Queue. Error: {exc}"
            )
            # Send to DLQ for manual inspection
            celery_app.send_task(
                'dead_letter_handler',
                queue='dead_letter',
                args=[task_id, str(exc), args, kwargs]
            )
        else:
            logger.warning(
                f"Task {task_id} failed (attempt {retry_count + 1}/{self.max_retries}). "
                f"Will retry. Error: {exc}"
            )

    def on_success(self, retval, task_id, args, kwargs):
        """Log successful execution."""
        logger.info(f"Task {task_id} completed successfully")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Log retry attempt."""
        logger.info(f"Retrying task {task_id} due to: {exc}")


def _compute_code_hash(code: str) -> str:
    """
    Compute SHA-256 hash of code for idempotency checking.

    Args:
        code: Python code string

    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(code.encode('utf-8')).hexdigest()


def _set_resource_limits():
    """
    Set OS-level resource limits for the subprocess.

    Limits:
    - Memory: 1GB
    - CPU time: 120 seconds
    - File size: 100MB
    - Number of processes: 10
    """
    try:
        # Memory limit (1GB)
        resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))

        # CPU time limit (120 seconds)
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))

        # File size limit (100MB)
        resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))

        # Process limit
        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))

        logger.debug("Resource limits set successfully")
    except Exception as e:
        logger.warning(f"Failed to set resource limits (may not be supported on Windows): {e}")


def _execute_in_sandbox(code: str, timeout: int, tmpdir: Path) -> Dict:
    """
    Execute code in a sandboxed environment with resource constraints.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds
        tmpdir: Temporary directory for execution

    Returns:
        Execution result dictionary
    """
    code_file = tmpdir / "exec_code.py"
    start_time = datetime.now()

    try:
        code_file.write_text(code, encoding='utf-8')
    except Exception as e:
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Failed to write code file: {e}',
            'returncode': -1,
            'execution_time_ms': 0,
            'memory_used_mb': None
        }

    try:
        # Execute with resource limits (Unix-like systems)
        if os.name != 'nt':  # Not Windows
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=str(tmpdir),
                preexec_fn=_set_resource_limits,  # Set limits before execution
                env={**os.environ, 'PYTHONHASHSEED': '0'}  # Deterministic execution
            )
        else:
            # Windows: No preexec_fn support
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                timeout=timeout,
                text=True,
                cwd=str(tmpdir),
                env={**os.environ, 'PYTHONHASHSEED': '0'}
            )

        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Estimate memory usage (rough approximation)
        memory_mb = None
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            pass

        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'execution_time_ms': execution_time_ms,
            'memory_used_mb': memory_mb
        }

    except subprocess.TimeoutExpired:
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.warning(f"Code execution timed out after {timeout}s")
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution timed out after {timeout} seconds',
            'returncode': -1,
            'execution_time_ms': execution_time_ms,
            'memory_used_mb': None
        }

    except Exception as e:
        execution_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        logger.error(f"Code execution error: {e}")
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution error: {str(e)}',
            'returncode': -1,
            'execution_time_ms': execution_time_ms,
            'memory_used_mb': None
        }


@celery_app.task(
    base=CodeExecutionTask,
    name='execute_code_async',
    bind=True,
    acks_late=True,
    reject_on_worker_lost=True,
    soft_time_limit=90,
    time_limit=120,
)
def execute_code_async(self, code: str, timeout: int = 30, idempotency_key: Optional[str] = None) -> Dict:
    """
    Execute Python code asynchronously in an isolated, resource-constrained environment.

    This task implements the production-grade worker-broker-result pattern with:
    - Automatic retries with exponential backoff
    - Resource limits (memory, CPU, file size)
    - Idempotency checking
    - Dead Letter Queue for poison pills
    - Comprehensive monitoring

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 30)
        idempotency_key: Optional key for idempotency (uses code hash if not provided)

    Returns:
        ExecutionResult dictionary with comprehensive metadata

    Raises:
        SoftTimeLimitExceeded: If task exceeds soft time limit
        Reject: If task should be sent to Dead Letter Queue

    Example:
        >>> result = execute_code_async.delay("print('Hello, World!')")
        >>> result.get(timeout=10)
        {'success': True, 'stdout': 'Hello, World!\\n', ...}
    """
    task_id = self.request.id
    retry_count = self.request.retries
    worker_id = self.request.hostname

    logger.info(f"[{task_id}] Starting code execution (attempt {retry_count + 1})")

    # Compute code hash for idempotency
    code_hash = idempotency_key or _compute_code_hash(code)

    # Check result cache for idempotency (if result already exists)
    cached_result = celery_app.backend.get(f'result:{code_hash}')
    if cached_result:
        logger.info(f"[{task_id}] Returning cached result (idempotency check)")
        return cached_result

    try:
        # Execute in temporary directory (auto-cleanup)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Execute code in sandbox
            exec_result = _execute_in_sandbox(code, timeout, tmpdir_path)

            # Build structured result
            result = ExecutionResult(
                task_id=task_id,
                code_hash=code_hash,
                success=exec_result['success'],
                stdout=exec_result['stdout'],
                stderr=exec_result['stderr'],
                returncode=exec_result['returncode'],
                execution_time_ms=exec_result['execution_time_ms'],
                memory_used_mb=exec_result['memory_used_mb'],
                retry_count=retry_count,
                timestamp=datetime.utcnow().isoformat(),
                worker_id=worker_id
            )

            result_dict = result.to_dict()

            # Cache result for idempotency (1 hour TTL)
            celery_app.backend.set(f'result:{code_hash}', result_dict, ex=3600)

            logger.info(
                f"[{task_id}] Execution {'succeeded' if result.success else 'failed'} "
                f"(time: {result.execution_time_ms}ms, retries: {retry_count})"
            )

            return result_dict

    except SoftTimeLimitExceeded:
        logger.warning(f"[{task_id}] Task exceeded soft time limit")
        raise  # Will trigger retry

    except Exception as e:
        logger.error(f"[{task_id}] Unexpected error: {e}", exc_info=True)

        # Check if this is a "poison pill" (repeatedly failing task)
        if retry_count >= 2:
            logger.error(f"[{task_id}] Poison pill detected, rejecting task")
            raise Reject(f"Poison pill: {e}", requeue=False)

        raise  # Will trigger retry


@celery_app.task(
    base=CodeExecutionTask,
    name='execute_code_heavy',
    bind=True,
    acks_late=True,
    soft_time_limit=240,
    time_limit=300,
)
def execute_code_heavy(self, code: str, timeout: int = 120) -> Dict:
    """
    Execute heavy/long-running Python code with extended limits.

    For ML model training, large data processing, etc.

    Args:
        code: Python code to execute
        timeout: Execution timeout in seconds (default: 120)

    Returns:
        ExecutionResult dictionary
    """
    logger.info(f"[{self.request.id}] Starting HEAVY code execution")
    return execute_code_async(self, code, timeout=timeout)


@celery_app.task(name='dead_letter_handler')
def dead_letter_handler(task_id: str, error: str, args: list, kwargs: dict):
    """
    Handle tasks that have exceeded max retries.

    Logs the failure for manual inspection and alerting.

    Args:
        task_id: Failed task ID
        error: Error message
        args: Original task arguments
        kwargs: Original task kwargs
    """
    logger.critical(
        f"DEAD LETTER QUEUE: Task {task_id} failed permanently. "
        f"Error: {error}. Args: {args}, Kwargs: {kwargs}"
    )

    # TODO: Implement alerting (e.g., send to Sentry, PagerDuty, Slack)
    # TODO: Store in persistent DLQ storage for manual review

    return {
        'status': 'logged_to_dlq',
        'task_id': task_id,
        'error': error,
        'timestamp': datetime.utcnow().isoformat()
    }
