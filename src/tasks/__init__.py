"""
Asynchronous task execution using Celery.
Production-grade code execution with proper isolation and monitoring.
"""

from src.tasks.celery_app import celery_app
from src.tasks.code_execution_tasks import execute_code_async, ExecutionResult

__all__ = ['celery_app', 'execute_code_async', 'ExecutionResult']
