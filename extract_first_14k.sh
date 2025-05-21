#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="first_14k_chars.txt"

if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: File '$INPUT_FILE' not found"
  exit 1
fi

# Extract first 14000 bytes (approximation for 14000 characters)
head -c 14000 "$INPUT_FILE" > "$OUTPUT_FILE"

# Count quotes in the extracted file
QUOTE_COUNT=$(grep -o '"' "$OUTPUT_FILE" | wc -l)

echo "Extracted first 14,000 characters to $OUTPUT_FILE"
echo "Found approximately $QUOTE_COUNT quote characters in the extract"
echo "View the file with: less $OUTPUT_FILE"