#!/bin/bash
#
# Start Celery workers for production-grade code execution
#
# Usage:
#   ./scripts/start_workers.sh [mode]
#
# Modes:
#   dev      - Single worker for development
#   prod     - Multiple workers with autoscaling
#   monitor  - Start Flower monitoring dashboard only
#

set -e

MODE=${1:-dev}

echo "================================"
echo "Hybrid Agentic System - Celery Workers"
echo "================================"
echo ""

# Check Redis connection
echo "Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running!"
    echo "Start Redis with: docker run -d -p 6379:6379 redis:alpine"
    echo "Or: docker-compose up -d redis"
    exit 1
fi
echo "✓ Redis is running"
echo ""

case $MODE in
    dev)
        echo "Starting DEVELOPMENT mode (single worker)..."
        celery -A src.tasks.celery_app worker \
            --loglevel=info \
            --queue=code_execution,code_execution_heavy \
            --concurrency=2 \
            --max-tasks-per-child=100 \
            --hostname=worker-dev@%h
        ;;

    prod)
        echo "Starting PRODUCTION mode (multiple workers with autoscaling)..."
        echo ""
        echo "Starting workers in background..."

        # Standard worker (autoscaling 2-8)
        celery -A src.tasks.celery_app worker \
            --loglevel=info \
            --queue=code_execution \
            --autoscale=8,2 \
            --max-tasks-per-child=100 \
            --hostname=worker-standard@%h \
            --detach \
            --logfile=outputs/logs/celery-worker-standard.log \
            --pidfile=outputs/pids/celery-worker-standard.pid

        echo "✓ Standard worker started (autoscale 2-8)"

        # Heavy worker (autoscaling 1-4)
        celery -A src.tasks.celery_app worker \
            --loglevel=info \
            --queue=code_execution_heavy \
            --autoscale=4,1 \
            --max-tasks-per-child=50 \
            --hostname=worker-heavy@%h \
            --detach \
            --logfile=outputs/logs/celery-worker-heavy.log \
            --pidfile=outputs/pids/celery-worker-heavy.pid

        echo "✓ Heavy worker started (autoscale 1-4)"

        # Start Flower monitoring
        celery -A src.tasks.celery_app flower \
            --port=5555 \
            --basic_auth=admin:admin123 \
            --detach \
            --logfile=outputs/logs/celery-flower.log \
            --pidfile=outputs/pids/celery-flower.pid

        echo "✓ Flower dashboard started at http://localhost:5555 (user: admin, pass: admin123)"
        echo ""
        echo "Workers started successfully!"
        echo ""
        echo "Monitor workers:"
        echo "  - Flower: http://localhost:5555"
        echo "  - Logs: tail -f outputs/logs/celery-*.log"
        echo ""
        echo "Stop workers:"
        echo "  - pkill -F outputs/pids/celery-worker-standard.pid"
        echo "  - pkill -F outputs/pids/celery-worker-heavy.pid"
        echo "  - pkill -F outputs/pids/celery-flower.pid"
        ;;

    monitor)
        echo "Starting Flower monitoring dashboard..."
        celery -A src.tasks.celery_app flower \
            --port=5555 \
            --basic_auth=admin:admin123

        ;;

    *)
        echo "Invalid mode: $MODE"
        echo ""
        echo "Usage: $0 [mode]"
        echo ""
        echo "Modes:"
        echo "  dev      - Single worker for development"
        echo "  prod     - Multiple workers with autoscaling"
        echo "  monitor  - Start Flower monitoring dashboard only"
        exit 1
        ;;
esac
