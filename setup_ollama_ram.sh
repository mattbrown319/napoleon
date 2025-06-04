#!/bin/bash

# This script sets up Ollama to load multiple copies of the model into RAM
# for better utilization of available memory and increased throughput

echo "Setting up Ollama for maximum RAM utilization..."

# Stop any existing Ollama processes very aggressively
echo "Stopping all existing Ollama processes..."
pkill -9 -f ollama || true
sleep 3

# Release all ports we might use
for PORT in 11434 11435 11436 11437; do
  echo "Releasing port $PORT if in use..."
  lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
done
sleep 2

# Double-check that ports are free
if lsof -ti :11435 >/dev/null; then
  echo "Error: Port 11435 is still in use. Please free this port manually."
  exit 1
fi

# Determine total system RAM and allocate 80% of it for Ollama
TOTAL_RAM_KB=$(sysctl hw.memsize | awk '{print $2/1024}')
TOTAL_RAM_GB=$(echo "scale=1; $TOTAL_RAM_KB/1024/1024" | bc)
OLLAMA_RAM_GB=$(echo "scale=0; $TOTAL_RAM_GB*0.8/1" | bc)

echo "System has approximately ${TOTAL_RAM_GB}GB RAM"
echo "Allocating ${OLLAMA_RAM_GB}GB for Ollama model instances"

# Calculate how many model copies we can fit in RAM
# Each qwen model is ~25GB in memory
MODEL_SIZE_GB=25
NUM_INSTANCES=$(echo "scale=0; $OLLAMA_RAM_GB/$MODEL_SIZE_GB" | bc)
echo "Can support approximately $NUM_INSTANCES model instances in RAM"

# Set reasonable limit to max 8 instances
if [ $NUM_INSTANCES -gt 8 ]; then
  NUM_INSTANCES=8
  echo "Limiting to $NUM_INSTANCES instances for stability"
fi

# Ensure we have at least 1 instance
if [ $NUM_INSTANCES -lt 1 ]; then
  NUM_INSTANCES=1
  echo "Setting minimum of 1 instance"
fi

# Start Ollama with aggressive memory settings for parallel processing
echo "Starting Ollama configured for $NUM_INSTANCES model instances with aggressive memory settings..."

# Set environment variables to control Ollama's memory usage - more aggressive settings
export OLLAMA_HOST=127.0.0.1:11435  # Use port 11435 instead of default
export OLLAMA_MMAP_THRESHOLD=1MB    # Force memory mapping for almost everything
export OLLAMA_MMAP_MULTIPLIER=$NUM_INSTANCES  # Allow multiple model copies in RAM
export OLLAMA_NUM_PARALLEL=$NUM_INSTANCES     # Enable parallel processing
export OLLAMA_KEEP_ALIVE=1h         # Keep models loaded longer

# Start a single Ollama server with these settings
nohup ollama serve > ~/ollama_mmap.log 2>&1 &
OLLAMA_PID=$!

echo "Waiting for Ollama to start..."
sleep 5

# Check if Ollama is running
if ps -p $OLLAMA_PID > /dev/null; then
  echo "Ollama is running with PID $OLLAMA_PID"
  echo "Listening on port 11435 (via OLLAMA_HOST environment variable)"
else
  echo "Failed to start Ollama, check log at ~/ollama_mmap.log"
  cat ~/ollama_mmap.log
  exit 1
fi

# Preload the model to make sure it's available
echo "Preloading model to ensure it's ready..."
curl -s "http://localhost:11435/api/tags" > /dev/null

echo -e "\nOllama is now configured with aggressive settings to load up to $NUM_INSTANCES copies of the model into RAM."
echo "Memory settings:"
echo "  OLLAMA_MMAP_THRESHOLD: 1MB (very aggressive)"
echo "  OLLAMA_MMAP_MULTIPLIER: $NUM_INSTANCES"
echo "  OLLAMA_NUM_PARALLEL: $NUM_INSTANCES"
echo ""
echo "When you run your script, it will now be able to utilize more of your available RAM."
echo "You should see RAM usage increase as parallel requests are processed."
echo ""
echo "When ready, run your script with:"
echo "python code/generate_qa_pairs.py napoleon_text.txt -o napoleon_index_test3 --port 11435 --workers $NUM_INSTANCES --verbose" 