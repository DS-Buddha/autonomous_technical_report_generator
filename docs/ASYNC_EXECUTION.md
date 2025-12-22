# Production-Grade Asynchronous Code Execution

## 🎯 Overview

The hybrid agentic system now supports **production-grade asynchronous code execution** using Celery workers with proper isolation, retry logic, and monitoring.

This implementation follows enterprise best practices for running untrusted/LLM-generated code safely.

---

## 🏗️ Architecture

### Worker-Broker-Result Pattern

```
API Server (FastAPI)
    ↓ submits task
Redis Broker (Queue)
    ↓ picks up task
Celery Worker (Isolated)
    ↓ executes code
    ↓ writes result
Redis Result Backend
    ↑ retrieves result
API Server (returns to user)
```

### Key Components

1. **Broker**: Redis - Manages task queue
2. **Worker**: Celery - Executes code in isolation
3. **Result Backend**: Redis - Stores execution results
4. **Dead Letter Queue**: Handles failed tasks

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install celery redis
```

### 2. Start Redis (Broker + Result Backend)

#### Using Docker:
```bash
docker run -d -p 6379:6379 redis:alpine
```

#### Using Docker Compose:
```bash
# See docker-compose.yml
docker-compose up -d redis
```

### 3. Start Celery Workers

```bash
# Standard worker
celery -A src.tasks.celery_app worker --loglevel=info --queue=code_execution

# Heavy workload worker (for ML tasks, etc.)
celery -A src.tasks.celery_app worker --loglevel=info --queue=code_execution_heavy --concurrency=2
```

### 4. (Optional) Start Celery Flower (Monitoring Dashboard)

```bash
celery -A src.tasks.celery_app flower --port=5555
```

Access dashboard at: http://localhost:5555

---

## 📝 Usage Examples

### Synchronous Execution (Legacy Mode)

```python
from src.tools.code_tools import CodeTools

code = """
import numpy as np
print(np.array([1, 2, 3]).mean())
"""

# Execute directly (blocks until complete)
result = CodeTools.execute_code(code, async_mode=False)

print(result['stdout'])  # Output: 2.0
print(result['success'])  # True
```

### Asynchronous Execution (Production Mode)

```python
from src.tools.code_tools import CodeTools

code = """
import time
for i in range(5):
    print(f"Processing {i}...")
    time.sleep(1)
"""

# Submit to worker and wait for result
result = CodeTools.execute_code(
    code,
    async_mode=True,
    wait_for_result=True
)

print(result['task_id'])  # e.g., "a1b2c3d4-..."
print(result['execution_time_ms'])  # e.g., 5234
print(result['stdout'])
```

### Fire-and-Forget (Non-Blocking)

```python
# Submit task and return immediately
result = CodeTools.execute_code(
    code,
    async_mode=True,
    wait_for_result=False
)

task_id = result['task_id']
print(f"Task submitted: {task_id}")

# Later, retrieve result
import time
time.sleep(10)
final_result = CodeTools.get_task_result(task_id, timeout=30)
print(final_result['stdout'])
```

---

## 🛡️ Safety Features

### 1. Resource Limits (Unix/Linux)

Each task runs with OS-level resource constraints:

- **Memory**: 1GB hard limit
- **CPU Time**: 120 seconds
- **File Size**: 100MB
- **Processes**: 10 max

### 2. The 3-Tier Timeout Rule

1. **Task Timeout** (soft): 90 seconds - Task receives `SoftTimeLimitExceeded`
2. **Task Timeout** (hard): 120 seconds - Worker kills task with `SIGTERM`
3. **OS Timeout**: Kernel kills subprocess if needed

### 3. Automatic Retries with Exponential Backoff

- Max retries: 3
- Backoff: 5s → 25s → 125s (with jitter)
- Retry on: `TimeoutExpired`, `MemoryError`, `OSError`

### 4. Dead Letter Queue (DLQ)

Tasks that fail 3 times are routed to DLQ for manual inspection.

```python
# View DLQ in Flower dashboard
# Or query Redis directly
redis-cli LRANGE dead_letter 0 -1
```

### 5. Idempotency

Tasks with the same code generate the same `code_hash`. Results are cached for 1 hour.

```python
# First execution
result1 = CodeTools.execute_code("print('Hello')", async_mode=True)

# Second execution (returns cached result instantly)
result2 = CodeTools.execute_code("print('Hello')", async_mode=True)

assert result1['code_hash'] == result2['code_hash']
```

---

## 📊 Monitoring & Observability

### Celery Flower Dashboard

Access at: http://localhost:5555

**Metrics:**
- Active workers
- Task throughput
- Queue depth
- Failure rates
- Execution times

### Key Metrics to Monitor

#### 1. Queue Depth
```bash
celery -A src.tasks.celery_app inspect active_queues
```

**Alert if**: Queue depth > 100 for >5 minutes

#### 2. Wait Time (Critical!)
```bash
# Time from task submission to start
# If increasing → scale workers, not API
```

#### 3. Worker Health
```bash
celery -A src.tasks.celery_app inspect ping
```

### Logging

All executions logged with structured data:

```python
{
    "task_id": "abc123",
    "code_hash": "sha256...",
    "success": true,
    "execution_time_ms": 1234,
    "memory_used_mb": 45.2,
    "retry_count": 0,
    "worker_id": "celery@worker1"
}
```

---

## ⚠️ Common Mistakes & Solutions

### Mistake 1: Wait-and-Block Anti-pattern

❌ **Wrong**:
```python
# API endpoint blocks for 2 minutes
result = execute_code(heavy_ml_code, async_mode=False)
```

✅ **Correct**:
```python
# API returns immediately
task = execute_code(heavy_ml_code, async_mode=True, wait_for_result=False)
return {"task_id": task['task_id'], "status": "processing"}
```

### Mistake 2: Dependency Entanglement

❌ **Wrong**: Using same Docker image for API + Worker

✅ **Correct**: Separate images
```dockerfile
# api.Dockerfile (lightweight)
FROM python:3.11-slim
RUN pip install fastapi uvicorn

# worker.Dockerfile (with ML libs)
FROM python:3.11
RUN pip install torch transformers celery redis
```

### Mistake 3: Not Monitoring Queue Depth

❌ **Wrong**: Only monitoring success/failure rates

✅ **Correct**: Monitor queue depth and wait time
```python
# Alert if queue depth > threshold
if queue_depth > 100:
    trigger_autoscale()
```

---

## 🔧 Production Deployment

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    command: celery -A src.tasks.celery_app worker --loglevel=info
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - redis
    deploy:
      replicas: 3  # Scale workers

  flower:
    build:
      context: .
      dockerfile: Dockerfile.worker
    command: celery -A src.tasks.celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis_data:
```

### Kubernetes Deployment

```yaml
# celery-worker-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: celery-worker
spec:
  replicas: 5
  template:
    spec:
      containers:
      - name: worker
        image: your-registry/hybrid-agentic-worker:latest
        command: ["celery", "-A", "src.tasks.celery_app", "worker"]
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: CELERY_BROKER_URL
          value: "redis://redis-service:6379/0"
```

### Autoscaling (K8s HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: celery-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: celery-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: External
    external:
      metric:
        name: celery_queue_depth
      target:
        type: AverageValue
        averageValue: "50"  # Scale if queue > 50 tasks per worker
```

---

## 🧪 Testing

### Unit Test

```python
def test_async_execution():
    code = "print('Hello, World!')"

    result = CodeTools.execute_code(code, async_mode=True)

    assert result['success'] is True
    assert 'Hello, World!' in result['stdout']
    assert result['task_id'] is not None
    assert result['execution_time_ms'] < 5000
```

### Load Test

```bash
# Using locust
locust -f tests/load/test_code_execution.py --host=http://localhost:8000
```

---

## 📈 Performance Characteristics

### Latency

- **Sync mode**: ~50-200ms overhead
- **Async mode**: ~100-500ms overhead (includes queue + scheduling)

### Throughput

- **Sync mode**: Limited by API server concurrency
- **Async mode**: Scales linearly with worker count

### Resource Usage

- **Worker memory**: ~200MB baseline + code execution
- **Redis memory**: ~10KB per queued task

---

## 🎓 Staff Engineer Lessons

### The "Poison Pill" Problem

**Symptom**: One bad task crashes workers in a loop

**Solution**: Implement max retries + DLQ

```python
# After 3 retries, send to DLQ
if retry_count >= 3:
    send_to_dlq(task)
    alert_ops_team()
```

### The "Disk Full" Outage

**Symptom**: Workers fill `/tmp`, node crashes

**Solution**: Aggressive cleanup + monitoring

```python
# Automatic cleanup
with tempfile.TemporaryDirectory() as tmpdir:
    # Code runs here
    pass  # tmpdir deleted automatically
```

### The "Side-Effect" Ban

**Rule**: Workers CANNOT write to production database

**Reason**: Prevents accidental data corruption from buggy code

```python
# ✅ Correct: Return data
return {"result": processed_data}

# ❌ Wrong: Write to DB
db.users.update(...)  # FORBIDDEN
```

---

## 🔗 Related Documentation

- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html#best-practices)
- [Redis Persistence](https://redis.io/topics/persistence)
- [Kubernetes Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

---

**Questions? Issues?**
- GitHub: [Create Issue](https://github.com/DS-Buddha/autonomous_technical_report_generator/issues)
- Logs: Check `outputs/app.log` and Flower dashboard
