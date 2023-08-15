#!/bin/bash

echo "Choose a scan mode:"
echo "1) Quick Python Scan (quick scan of python files)"
echo "2) Full Python Scan (full scan of python files)"
echo "3) Extended Full Scan (full scan of Python files plus listing of all files and directories)"
read -p "Enter your choice (1/2/3): " choice

echo "Press Enter to start..."
read

if [[ $choice -eq 1 ]]; then
    pattern='^class |^def '
    find . -name "*.py" -type f -print0 | while IFS= read -r -d $'\0' file; do
        echo "$file"
        grep -E "$pattern" "$file"
        echo
    done

elif [[ $choice -eq 2 ]]; then
    pattern='^class |^def |^@|^[[:space:]]{4}def |^[[:space:]]{4}@'
    find . -name "*.py" -type f -print0 | while IFS= read -r -d $'\0' file; do
        echo "$file"
        grep -E "$pattern" "$file"
        echo
    done

elif [[ $choice -eq 3 ]]; then
    pattern='^class |^def |^@|^[[:space:]]{4}def |^[[:space:]]{4}@'
    # Display all files and directories but ignore hidden ones
    find . -not -path './.*' -and -not -path '*/.*' -print0 | while IFS= read -r -d $'\0' file_or_dir; do
        # Check if the file is a Python file
        if [[ $file_or_dir == *.py ]]; then
            echo "$file_or_dir"
            grep -E "$pattern" "$file_or_dir"
            echo
        else
            echo "$file_or_dir"
            echo
        fi
    done
else
    echo "Invalid choice."
fi

echo "Press Enter to end..."
read
