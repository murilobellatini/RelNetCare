#!/bin/bash

find . -name "*.py" -type f -print0 | while IFS= read -r -d $'\0' file; do
    echo "$file"
    grep -E '^class |^def ' "$file"
    echo
done
