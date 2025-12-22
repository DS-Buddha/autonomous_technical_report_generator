"""
Example: Production-Grade Asynchronous Code Execution

Demonstrates:
1. Synchronous vs Asynchronous execution
2. Fire-and-forget pattern
3. Batch execution
4. Error handling and retries
5. Monitoring task status
"""

import time
from src.tools.code_tools import CodeTools


def example_1_sync_vs_async():
    """Compare synchronous and asynchronous execution."""
    print("=" * 60)
    print("Example 1: Synchronous vs Asynchronous Execution")
    print("=" * 60)

    code = """
import time
print("Starting computation...")
time.sleep(2)
print("Computation complete!")
result = sum(range(1000000))
print(f"Result: {result}")
"""

    # Synchronous execution (blocks)
    print("\n1a. SYNCHRONOUS EXECUTION:")
    print("   Starting... (will block for 2+ seconds)")
    start = time.time()
    result_sync = CodeTools.execute_code(code, async_mode=False)
    duration_sync = time.time() - start
    print(f"   ✓ Completed in {duration_sync:.2f}s")
    print(f"   Output: {result_sync['stdout'].strip()}")

    # Asynchronous execution (also blocks, but uses worker)
    print("\n1b. ASYNCHRONOUS EXECUTION (blocking):")
    print("   Starting... (submitted to worker)")
    start = time.time()
    result_async = CodeTools.execute_code(code, async_mode=True, wait_for_result=True)
    duration_async = time.time() - start
    print(f"   ✓ Completed in {duration_async:.2f}s")
    print(f"   Task ID: {result_async['task_id']}")
    print(f"   Execution time: {result_async['execution_time_ms']}ms")
    print(f"   Output: {result_async['stdout'].strip()}")

    print("\n")


def example_2_fire_and_forget():
    """Fire-and-forget pattern for non-blocking execution."""
    print("=" * 60)
    print("Example 2: Fire-and-Forget (Non-Blocking)")
    print("=" * 60)

    code = """
import time
for i in range(5):
    print(f"Processing step {i+1}/5...")
    time.sleep(1)
print("All steps complete!")
"""

    # Submit task without waiting
    print("\n2a. Submitting task (non-blocking)...")
    result = CodeTools.execute_code(code, async_mode=True, wait_for_result=False)

    task_id = result['task_id']
    print(f"   ✓ Task submitted: {task_id}")
    print(f"   Status: {result['status']}")
    print("   API call returned immediately!")

    # Do other work while task runs
    print("\n2b. Doing other work while task runs in background...")
    for i in range(3):
        print(f"   Main thread: working on task {i+1}...")
        time.sleep(1)

    # Retrieve result
    print("\n2c. Retrieving task result...")
    final_result = CodeTools.get_task_result(task_id, timeout=30)

    if final_result and final_result.get('success'):
        print(f"   ✓ Task completed successfully!")
        print(f"   Execution time: {final_result['execution_time_ms']}ms")
        print(f"   Output:\n{final_result['stdout']}")
    else:
        print(f"   ✗ Task failed or timed out")
        if final_result:
            print(f"   Error: {final_result.get('stderr', 'Unknown error')}")

    print("\n")


def example_3_batch_execution():
    """Execute multiple tasks in parallel."""
    print("=" * 60)
    print("Example 3: Batch Execution (Parallel)")
    print("=" * 60)

    # Create 5 different tasks
    tasks = []
    for i in range(5):
        code = f"""
import time
import random
sleep_time = random.uniform(1, 3)
time.sleep(sleep_time)
print(f"Task {i+1} completed after {{sleep_time:.2f}}s")
"""
        print(f"\n3{chr(97+i)}. Submitting task {i+1}...")
        result = CodeTools.execute_code(code, async_mode=True, wait_for_result=False)
        tasks.append({
            'id': i+1,
            'task_id': result['task_id'],
            'status': 'pending'
        })
        print(f"   ✓ Task ID: {result['task_id']}")

    print(f"\n   All {len(tasks)} tasks submitted!")

    # Wait for all tasks to complete
    print("\n   Waiting for tasks to complete...")
    time.sleep(5)  # Give tasks time to execute

    # Retrieve results
    print("\n   Results:")
    for task in tasks:
        result = CodeTools.get_task_result(task['task_id'], timeout=10)
        if result and result.get('success'):
            task['status'] = 'completed'
            print(f"   ✓ Task {task['id']}: {result['stdout'].strip()}")
        else:
            task['status'] = 'failed'
            print(f"   ✗ Task {task['id']}: Failed")

    completed = sum(1 for t in tasks if t['status'] == 'completed')
    print(f"\n   Summary: {completed}/{len(tasks)} tasks completed")

    print("\n")


def example_4_error_handling():
    """Demonstrate error handling and retries."""
    print("=" * 60)
    print("Example 4: Error Handling and Retries")
    print("=" * 60)

    # Code with error
    code_with_error = """
import sys
print("Starting problematic code...")
x = 1 / 0  # This will raise ZeroDivisionError
print("This line won't execute")
"""

    print("\n4a. Executing code with error (async)...")
    result = CodeTools.execute_code(code_with_error, async_mode=True)

    if result['success']:
        print("   ✓ Execution succeeded")
    else:
        print(f"   ✗ Execution failed (as expected)")
        print(f"   Return code: {result['returncode']}")
        print(f"   Error: {result['stderr'][:200]}")
        print(f"   Retry count: {result.get('retry_count', 0)}")

    # Code that times out
    code_timeout = """
import time
print("Starting long computation...")
time.sleep(300)  # Will timeout
"""

    print("\n4b. Executing code that times out (async, 5s timeout)...")
    result = CodeTools.execute_code(code_timeout, async_mode=True, timeout=5)

    if result['success']:
        print("   ✓ Execution succeeded")
    else:
        print(f"   ✗ Execution timed out (as expected)")
        print(f"   Error: {result['stderr'][:200]}")

    print("\n")


def example_5_idempotency():
    """Demonstrate idempotency (same code returns cached result)."""
    print("=" * 60)
    print("Example 5: Idempotency (Result Caching)")
    print("=" * 60)

    code = """
import random
import time
time.sleep(1)
random_number = random.randint(1, 1000000)
print(f"Random number: {random_number}")
"""

    print("\n5a. First execution (will compute)...")
    start = time.time()
    result1 = CodeTools.execute_code(code, async_mode=True)
    duration1 = time.time() - start

    print(f"   ✓ Execution time: {duration1:.2f}s")
    print(f"   Task ID: {result1['task_id']}")
    print(f"   Code hash: {result1['code_hash']}")
    print(f"   Output: {result1['stdout'].strip()}")

    print("\n5b. Second execution (should return cached result)...")
    start = time.time()
    result2 = CodeTools.execute_code(code, async_mode=True)
    duration2 = time.time() - start

    print(f"   ✓ Execution time: {duration2:.2f}s (much faster!)")
    print(f"   Task ID: {result2['task_id']}")
    print(f"   Code hash: {result2['code_hash']}")
    print(f"   Output: {result2['stdout'].strip()}")

    if result1['code_hash'] == result2['code_hash']:
        print("\n   ✓ Code hashes match - idempotency confirmed!")
        print(f"   Speedup: {duration1/duration2:.2f}x faster")
    else:
        print("\n   ✗ Code hashes don't match - unexpected!")

    print("\n")


def main():
    """Run all examples."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Production-Grade Async Code Execution Examples           ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print("\n")

    try:
        # Run examples
        example_1_sync_vs_async()
        example_2_fire_and_forget()
        example_3_batch_execution()
        example_4_error_handling()
        example_5_idempotency()

        print("=" * 60)
        print("✓ All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()

    print("\n")


if __name__ == "__main__":
    main()
