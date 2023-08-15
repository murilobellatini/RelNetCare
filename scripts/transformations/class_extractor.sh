#!/bin/bash

echo "Choose a scan mode:"
echo "1) Full Scan (includes methods and decorators)"
echo "2) Quick Scan (only class and function names)"
read -p "Enter your choice (1/2): " choice

echo "Press Enter to start..."
read

if [[ $choice -eq 1 ]]; then
    pattern='^class |^def |^@|^[[:space:]]{4}def |^[[:space:]]{4}@'
else
    pattern='^class |^def '
fi

find . -name "*.py" -type f -print0 | while IFS= read -r -d $'\0' file; do
    echo "$file"
    grep -E "$pattern" "$file"
    echo
done

echo "Press Enter to end..."
read