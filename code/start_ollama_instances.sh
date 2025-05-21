#!/bin/bash
# Start multiple Ollama instances on different ports

# Default ports to use
PORTS=(11434 11435 11436 11437)

# Usage information
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --ports PORT1,PORT2,...   Comma-separated list of ports (default: 11434,11435,11436,11437)"
    echo "  -h, --help                    Show this help message"
    exit 1
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--ports)
            IFS=',' read -ra PORTS <<< "$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check if ports are already in use
for PORT in "${PORTS[@]}"; do
    if lsof -i :$PORT > /dev/null 2>&1; then
        echo "Port $PORT is already in use"
    fi
done

# Start Ollama instances
for PORT in "${PORTS[@]}"; do
    if [[ $PORT -eq 11434 ]]; then
        echo "Checking if default Ollama instance is running on port 11434..."
        if ! lsof -i :11434 > /dev/null 2>&1; then
            echo "Starting default Ollama instance on port 11434..."
            ollama serve > ollama_11434.log 2>&1 &
            echo "Started Ollama on port 11434 (PID: $!)"
        else
            echo "Default Ollama instance already running on port 11434"
        fi
    else
        echo "Starting Ollama instance on port $PORT..."
        OLLAMA_HOST=127.0.0.1:$PORT ollama serve > ollama_$PORT.log 2>&1 &
        echo "Started Ollama on port $PORT (PID: $!)"
    fi
done

echo ""
echo "Started Ollama instances on ports: ${PORTS[*]}"
echo "To use these instances with the hierarchical analyzer, run:"
echo "python code/test_hierarchical.py --ports $(IFS=,; echo "${PORTS[*]}") path/to/file.txt"
echo ""
echo "To stop all Ollama instances, use: pkill -f 'ollama serve'"
echo "" 