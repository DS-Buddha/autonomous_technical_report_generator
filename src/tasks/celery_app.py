"""
Celery application configuration for asynchronous code execution.

Production-grade configuration with:
- Task routing and prioritization
- Result backend for persistence
- Monitoring and health checks
- Resource limits and retry policies
"""

import os
from celery import Celery
from kombu import Queue, Exchange
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Celery broker configuration (Redis)
BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

# Create Celery app
celery_app = Celery(
    'hybrid_agentic_system',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=['src.tasks.code_execution_tasks']
)

# Celery Configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        'src.tasks.code_execution_tasks.execute_code_async': {
            'queue': 'code_execution',
            'routing_key': 'code.execute',
        },
        'src.tasks.code_execution_tasks.execute_code_heavy': {
            'queue': 'code_execution_heavy',
            'routing_key': 'code.execute.heavy',
        },
    },

    # Queue configuration with priorities
    task_queues=(
        Queue(
            'code_execution',
            Exchange('code_execution'),
            routing_key='code.execute',
            priority=5,
        ),
        Queue(
            'code_execution_heavy',
            Exchange('code_execution_heavy'),
            routing_key='code.execute.heavy',
            priority=10,  # Higher priority for heavy tasks
        ),
        Queue(
            'dead_letter',  # Dead Letter Queue for failed tasks
            Exchange('dead_letter'),
            routing_key='dead_letter',
        ),
    ),

    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,  # Persist results to backend
    result_extended=True,  # Store additional metadata

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for isolation
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks
    worker_disable_rate_limits=False,

    # Task acknowledgment
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Requeue on worker crash

    # Monitoring and logging
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,

    # Resource limits (enforced at task level)
    task_time_limit=300,  # Hard limit: 5 minutes
    task_soft_time_limit=240,  # Soft limit: 4 minutes

    # Retry policy defaults
    task_autoretry_for=(Exception,),
    task_max_retries=3,
    task_default_retry_delay=60,  # Retry after 60 seconds

    # Dead Letter Queue configuration
    task_reject_on_exc=True,  # Send to DLQ on exception
)

# Task annotations for per-task configuration
celery_app.conf.task_annotations = {
    'src.tasks.code_execution_tasks.execute_code_async': {
        'rate_limit': '10/m',  # 10 tasks per minute
        'time_limit': 120,  # 2 minutes hard limit
        'soft_time_limit': 90,  # 1.5 minutes soft limit
    },
    'src.tasks.code_execution_tasks.execute_code_heavy': {
        'rate_limit': '5/m',  # 5 tasks per minute
        'time_limit': 300,  # 5 minutes hard limit
        'soft_time_limit': 240,  # 4 minutes soft limit
    },
}

logger.info("Celery app configured successfully")
logger.info(f"Broker: {BROKER_URL}")
logger.info(f"Result backend: {RESULT_BACKEND}")


# Health check task
@celery_app.task(name='health_check')
def health_check():
    """Simple health check task for monitoring."""
    return {'status': 'healthy', 'worker': 'responsive'}


# Periodic task cleanup (optional - requires celery beat)
@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks for maintenance."""
    # Clean up old results every hour
    sender.add_periodic_task(
        3600.0,
        cleanup_old_results.s(),
        name='cleanup old results'
    )


@celery_app.task(name='cleanup_old_results')
def cleanup_old_results():
    """Clean up old task results from backend."""
    try:
        # Implement cleanup logic here
        # For Redis backend, results auto-expire based on result_expires
        logger.info("Periodic cleanup executed")
        return {'status': 'cleanup_completed'}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {'status': 'cleanup_failed', 'error': str(e)}
