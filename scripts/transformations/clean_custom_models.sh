#!/bin/bash

echo ""
echo "# GLOBAL DISK USAGE"
df -h . /mnt/vdb1

echo ""
echo "# USER DISTRIBUTION"
du -h --max-depth=1 /mnt/vdb1 2>&1 | grep -v "Permission denied" | awk -v total=313 '{size=$1; sub(/[A-Za-z]/, "", size); printf("%s\t%.2f%%\n", $0, (size/total)*100)}'

# Path to be cleaned
TARGET_PATH="/mnt/vdb1/murilo/models/custom/llama-7B-hf-lora-adaptor/"

# Check if folder exists and is not empty
if [ -d "$TARGET_PATH" ] && [ "$(ls -A $TARGET_PATH)" ]; then
  # Count folders and calculate total size
  NUM_FOLDERS=$(find "$TARGET_PATH" -type d | wc -l)
  TOTAL_SIZE=$(du -sh "$TARGET_PATH" 2>/dev/null | cut -f1)

  # List folders to be deleted
  echo "Folders to be deleted:"
  find "$TARGET_PATH" -type d

  # Confirm with the user
  read -p "About to delete $NUM_FOLDERS folders from $TARGET_PATH totaling $TOTAL_SIZE. Are you sure? (y/N) " -n 1 -r
  echo # Newline

  if [[ $REPLY =~ ^[Yy]$ ]]
  then
    rm -rf "$TARGET_PATH"
    echo "Folders and their content deleted."
  else
    echo "Operation cancelled."
  fi
else
  echo "No folders found in $TARGET_PATH. Nothing to delete."
fi

echo ""
echo "# GLOBAL DISK USAGE"
df -h . /mnt/vdb1

echo ""
echo "# USER DISTRIBUTION"
du -h --max-depth=1 /mnt/vdb1 2>&1 | grep -v "Permission denied" | awk -v total=313 '{size=$1; sub(/[A-Za-z]/, "", size); printf("%s\t%.2f%%\n", $0, (size/total)*100)}'


