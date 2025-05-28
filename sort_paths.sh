#!/bin/bash

# Check input arguments
if [ "$#" -ne 2 ]; then
	echo "Usage: $0 <input_file> <output_file>"
	exit 1
fi

input_file="$1"
output_file="$2"

# Extract paths, sort, and write to output file
grep -oP 'Path: \K.*' "$input_file" | sort > "$output_file"

echo "Sorted paths written to $output_file"
