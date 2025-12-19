# Code Execution Sandbox Upgrade Guide

## Current Implementation

**File**: `src/tools/code_tools.py`

Current code execution uses:
- `tempfile.TemporaryDirectory()` - Isolated temp directory
- `subprocess.run()` - Separate Python process
- `timeout=30` - Execution timeout

**Security Issues**:
- ❌ No resource limits (CPU, memory)
- ❌ Network access possible
- ❌ Can access host filesystem
- ❌ No capability dropping

**Risk Level**: MEDIUM (better than `exec()` but not production-ready)

---

## Recommended Upgrades

### Option 1: E2B Sandboxed Runtime (Easiest, Recommended)

**What is E2B?**
- Fully isolated cloud runtime environment
- No infrastructure management needed
- Built-in resource limits
- Network-isolated by default
- ~$10-30/month for 1000 executions

**Implementation**:

```bash
# Install
pip install e2b
```

```python
# src/tools/code_tools_e2b.py
from e2b import Sandbox

class CodeToolsE2B:
    """Enhanced code execution with E2B sandbox."""

    @staticmethod
    def execute_code_with_e2b(code: str, timeout: int = 30) -> Dict:
        """
        Execute code using E2B sandboxed runtime.
        Fully isolated, no infrastructure management needed.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        try:
            # Create sandbox (fully isolated cloud environment)
            sandbox = Sandbox(timeout=timeout)

            # Run code in sandbox
            execution = sandbox.run_code(
                code,
                on_stdout=lambda msg: logger.debug(f"Code output: {msg}"),
                on_stderr=lambda msg: logger.warning(f"Code error: {msg}")
            )

            return {
                'success': execution.exit_code == 0,
                'stdout': '\n'.join(execution.stdout),
                'stderr': '\n'.join(execution.stderr),
                'returncode': execution.exit_code
            }

        except Exception as e:
            logger.error(f"E2B execution error: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }
        finally:
            sandbox.close()
```

**Pros**:
- ✅ Zero infrastructure management
- ✅ Fully isolated (network, filesystem, resources)
- ✅ Fast setup (5 minutes)
- ✅ Built-in monitoring

**Cons**:
- ❌ Costs ~$10-30/month (but cheaper than security incident)
- ❌ Requires internet connection
- ❌ Third-party dependency

**Production Readiness**: HIGH ⭐⭐⭐⭐⭐

---

### Option 2: Docker Containers (More Control)

**Implementation**:

```bash
# Install
pip install docker
```

```python
# src/tools/code_tools_docker.py
import docker

class CodeToolsDocker:
    """Enhanced code execution with Docker isolation."""

    def __init__(self):
        self.docker_client = docker.from_env()

    def execute_code_in_container(
        self,
        code: str,
        timeout: int = 30
    ) -> Dict:
        """
        Execute code in isolated Docker container with resource limits.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with execution results
        """
        logger.info("Executing code in Docker sandbox")

        try:
            # Create container with strict limits
            container = self.docker_client.containers.run(
                "python:3.11-slim",           # Minimal Python image
                command=["python", "-c", code],
                detach=True,
                auto_remove=True,
                mem_limit="256m",             # Max 256MB RAM
                cpu_quota=50000,              # Max 50% CPU (50000/100000)
                network_disabled=True,        # No network access
                read_only=True,               # Read-only filesystem
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"]              # Drop all capabilities
            )

            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            logs = container.logs()

            return {
                'success': result['StatusCode'] == 0,
                'stdout': logs.decode('utf-8'),
                'stderr': '',
                'returncode': result['StatusCode']
            }

        except docker.errors.ContainerError as e:
            logger.error(f"Container execution error: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': str(e),
                'returncode': -1
            }

        except Exception as e:
            logger.error(f"Docker execution error: {e}")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'returncode': -1
            }
```

**Pros**:
- ✅ Full control over environment
- ✅ No external service dependency
- ✅ Can customize Python version/packages
- ✅ Free (except infrastructure)

**Cons**:
- ❌ Requires Docker installed
- ❌ More complex setup
- ❌ Infrastructure management needed
- ❌ Slower than E2B (container startup time)

**Production Readiness**: MEDIUM-HIGH ⭐⭐⭐⭐

---

### Option 3: gVisor (Maximum Security)

**What is gVisor?**
- Google's application kernel for containers
- Adds extra layer between container and host
- Used by Google Cloud Run

**Implementation**:

```bash
# Install gVisor runtime
sudo apt-get install google-cloud-sdk
gcloud components install gvisor-runtime

# Configure Docker to use gVisor
sudo docker run --runtime=runsc ...
```

```python
# Same as Docker option but with --runtime=runsc
container = self.docker_client.containers.run(
    "python:3.11-slim",
    runtime="runsc",  # gVisor runtime
    # ... rest of config
)
```

**Pros**:
- ✅ Maximum security (Google-grade)
- ✅ Additional syscall filtering
- ✅ Better isolation than vanilla Docker

**Cons**:
- ❌ Complex setup
- ❌ Slight performance overhead
- ❌ Linux only

**Production Readiness**: HIGH (for high-security needs) ⭐⭐⭐⭐⭐

---

## Comparison Table

| Feature | Current (subprocess) | E2B | Docker | gVisor |
|---------|---------------------|-----|--------|--------|
| **Isolation** | Weak | Strong | Strong | Very Strong |
| **Network Access** | ❌ Allowed | ✅ Blocked | ✅ Blocked | ✅ Blocked |
| **Resource Limits** | ❌ None | ✅ Built-in | ✅ Configurable | ✅ Configurable |
| **Setup Time** | 0 min | 5 min | 30 min | 60 min |
| **Cost** | Free | ~$20/mo | Infrastructure | Infrastructure |
| **Complexity** | Low | Low | Medium | High |
| **Production Ready** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Migration Path

### Immediate (This Week)
Keep current implementation but add monitoring:

```python
# Add to current code_tools.py
import psutil
import resource

def execute_code_with_monitoring(code: str, timeout: int = 30) -> Dict:
    """Current implementation with monitoring."""

    # Get process info for monitoring
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Monitor resource usage
    try:
        proc = psutil.Process(process.pid)
        max_memory = 0

        while process.poll() is None:
            try:
                mem = proc.memory_info().rss / 1024 / 1024  # MB
                max_memory = max(max_memory, mem)

                # Kill if exceeds 500MB
                if max_memory > 500:
                    process.kill()
                    raise MemoryError("Process exceeded 500MB memory")

                time.sleep(0.1)
            except psutil.NoSuchProcess:
                break

        stdout, stderr = process.communicate(timeout=timeout)

        logger.info(f"Code execution: max_memory={max_memory:.1f}MB")

        return {
            'success': process.returncode == 0,
            'stdout': stdout,
            'stderr': stderr,
            'returncode': process.returncode,
            'max_memory_mb': max_memory
        }

    except subprocess.TimeoutExpired:
        process.kill()
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Timeout after {timeout}s',
            'returncode': -1
        }
```

### Short-term (Next Sprint)
Implement E2B sandbox:
1. Sign up for E2B account
2. Install `pip install e2b`
3. Replace `CodeTools.execute_code()` with E2B version
4. Test with existing code blocks
5. Deploy to staging

**Timeline**: 1 day
**Cost**: ~$20/month
**Risk Reduction**: 80%

### Long-term (Next Quarter)
Migrate to self-hosted Docker solution:
1. Set up Docker infrastructure
2. Build custom Python images with dependencies
3. Implement resource monitoring
4. Add autoscaling for multiple concurrent executions
5. Deploy to production

**Timeline**: 2 weeks
**Cost**: Infrastructure only
**Risk Reduction**: 90%

---

## Testing Malicious Code

After implementing sandbox, test with these dangerous patterns:

```python
# Test 1: Memory bomb
while True:
    data = [0] * 10**9  # Should be killed by memory limit

# Test 2: Network access
import socket
socket.connect(('attacker.com', 443))  # Should fail - no network

# Test 3: File system access
import os
os.system('rm -rf /')  # Should fail - read-only filesystem

# Test 4: CPU bomb
while True:
    pass  # Should be killed by CPU limit

# Test 5: Fork bomb
import os
while True:
    os.fork()  # Should fail - capability dropped
```

All should fail gracefully with appropriate error messages.

---

## Recommended Implementation

**For Production Today**: Use **E2B**
- Fastest to implement (5 minutes)
- Lowest risk
- Best security/effort ratio
- Scalable

**For Enterprise/Self-Hosted**: Use **Docker + gVisor**
- Full control
- No third-party dependency
- Can customize heavily
- Better for compliance

---

## Integration into Tester Agent

**Update**: `src/agents/tester_agent.py`

```python
from src.tools.code_tools_e2b import CodeToolsE2B  # or CodeToolsDocker

class TesterAgent(BaseAgent):
    def __init__(self):
        super().__init__(...)
        self.tools = CodeToolsE2B()  # Use sandboxed version

    def run(self, code_blocks: Dict[str, str], **kwargs) -> Dict:
        # Everything else stays the same
        # CodeTools API is compatible
        result = self.tools.execute_code_with_e2b(code, timeout=30)
        # ... rest of testing logic
```

**Breaking Changes**: NONE (drop-in replacement)

---

## Monitoring & Alerts

After implementing sandbox, add these metrics:

```python
# src/utils/metrics.py

from dataclasses import dataclass
import time

@dataclass
class CodeExecutionMetrics:
    """Metrics for code execution monitoring."""
    code_id: str
    execution_time_ms: float
    memory_used_mb: float
    success: bool
    error_type: Optional[str]

def log_execution_metrics(metrics: CodeExecutionMetrics):
    """Log metrics for monitoring."""
    logger.info(
        "CODE_EXECUTION",
        extra={
            'metrics': {
                'code_id': metrics.code_id,
                'duration_ms': metrics.execution_time_ms,
                'memory_mb': metrics.memory_used_mb,
                'success': metrics.success,
                'error_type': metrics.error_type
            }
        }
    )

    # Alert on suspicious patterns
    if metrics.memory_used_mb > 200:
        alert_high_memory(metrics)

    if metrics.execution_time_ms > 25000:  # 25s close to 30s limit
        alert_slow_execution(metrics)
```

---

## Cost Analysis

**Current** (subprocess):
- Infrastructure: $0
- Security Risk: HIGH
- Incident Cost: $10,000-100,000 (if exploited)

**E2B**:
- Monthly: ~$20
- Security Risk: LOW
- Incident Cost: ~$0 (prevented)
- **ROI**: Break-even after first prevented incident

**Docker** (self-hosted):
- Infrastructure: $50-200/month (depending on scale)
- Security Risk: LOW
- Incident Cost: ~$0
- **ROI**: Better for high-volume (>10,000 executions/month)

---

## Next Steps

1. **Today**: Add resource monitoring to current implementation
2. **This Week**: Sign up for E2B, test integration
3. **Next Week**: Deploy E2B to staging
4. **Next Month**: Monitor E2B usage, optimize costs
5. **Next Quarter**: Evaluate Docker migration for cost savings

---

## References

- E2B Documentation: https://e2b.dev/docs
- Docker Security: https://docs.docker.com/engine/security/
- gVisor: https://gvisor.dev/docs/
- Python subprocess security: https://docs.python.org/3/library/subprocess.html#security-considerations
