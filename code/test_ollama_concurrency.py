#!/usr/bin/env python3
"""
Test script to verify if Ollama can handle multiple concurrent requests.
This will send 5 simultaneous requests to Ollama and time their execution.
"""

import subprocess
import threading
import time
import concurrent.futures
import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger('ollama_concurrency_test')

def query_ollama(prompt, model, prompt_id):
    """Send a query to Ollama and measure execution time"""
    thread_id = threading.get_ident()
    start_time = time.time()
    
    logger.info(f"[{prompt_id}] Thread {thread_id}: Starting Ollama query")
    
    try:
        command = ["ollama", "run", model, prompt]
        
        result = subprocess.run(
            command,
            capture_output=True, text=True, timeout=240
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode != 0:
            logger.error(f"[{prompt_id}] Thread {thread_id}: Ollama command failed: {result.stderr}")
            return prompt_id, False, duration
        
        logger.info(f"[{prompt_id}] Thread {thread_id}: Ollama query completed in {duration:.2f} seconds")
        return prompt_id, True, duration
    
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"[{prompt_id}] Thread {thread_id}: Error querying Ollama: {str(e)}")
        return prompt_id, False, duration

def main():
    parser = argparse.ArgumentParser(description='Test Ollama concurrency')
    parser.add_argument('--model', help='Ollama model to use', default='qwen3:235b-a22b')
    parser.add_argument('--workers', type=int, help='Number of concurrent workers', default=10)
    args = parser.parse_args()
    
    model = args.model
    worker_count = args.workers
    
    logger.info(f"Testing Ollama concurrency with {worker_count} workers using model: {model}")
    
    # Create different prompts for each worker
    prompts = [
        f"Explain the concept of democracy in {i+1} sentences." 
        for i in range(worker_count)
    ]
    
    # Run queries concurrently
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for i, prompt in enumerate(prompts):
            futures.append(executor.submit(query_ollama, prompt, model, i+1))
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Analyze and report results
    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    
    logger.info("\n----- RESULTS -----")
    logger.info(f"Total time for {worker_count} requests: {total_duration:.2f} seconds")
    logger.info(f"Successful requests: {len(successful)}/{worker_count}")
    logger.info(f"Failed requests: {len(failed)}/{worker_count}")
    
    if successful:
        avg_time = sum(r[2] for r in successful) / len(successful)
        logger.info(f"Average request time: {avg_time:.2f} seconds")
        logger.info(f"Speedup factor: {sum(r[2] for r in successful) / total_duration:.2f}x")
    
    # True parallelism check
    if successful and len(successful) > 1:
        if total_duration < sum(r[2] for r in successful) / 2:
            logger.info("✅ PARALLELISM CONFIRMED: Total time is significantly less than sequential time")
        else:
            logger.info("❌ NO PARALLELISM DETECTED: Requests appear to be processed sequentially")
    
    if failed:
        logger.info("\nFailed request IDs: " + ", ".join(str(r[0]) for r in failed))

if __name__ == "__main__":
    main() 