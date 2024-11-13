#!/bin/bash

# Create output file
output_file="combined_project_files.txt"
touch "$output_file"

# Function to add a file with header
add_file_with_header() {
    local file=$1
    if [ -f "$file" ]; then
        echo -e "\n############################################" >> "$output_file"
        echo "# File: $file" >> "$output_file"
        echo "############################################" >> "$output_file"
        echo "" >> "$output_file"
        cat "$file" >> "$output_file"
        echo -e "\n" >> "$output_file"
    fi
}

# Clear the output file if it exists
> "$output_file"

# Add timestamp to the beginning
echo "# Project files combined on $(date)" >> "$output_file"
echo "# Working directory: $(pwd)" >> "$output_file"
echo -e "\n" >> "$output_file"

# First add requirements.txt if it exists
add_file_with_header "requirements.txt"

# Then add .env if it exists
add_file_with_header ".env"

# Find and add all Python files
find . -type f -name "*.py" | while read -r file; do
    add_file_with_header "$file"
done

# Print summary
echo "Files have been combined into $output_file"
echo "Summary of files processed:"
echo "----------------------------------------"
echo "Python files found:"
find . -type f -name "*.py" | sed 's/^/- /'
echo "----------------------------------------"
if [ -f "requirements.txt" ]; then
    echo "- requirements.txt"
fi
if [ -f ".env" ]; then
    echo "- .env"
fi