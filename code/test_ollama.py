import subprocess
import sys

# Just test if Ollama responds to a simple prompt
model = sys.argv[1] if len(sys.argv) > 1 else "qwen3:235b-a22b"
result = subprocess.run(
    ["ollama", "run", model, "Say hello"],
    capture_output=True, text=True, timeout=30
)
print(f"Return code: {result.returncode}")
print(f"Output: {result.stdout}") 