#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

SOURCE_DIR="${SCRIPT_DIR}/../cinnamon/testbench/generated/"

TARGET_DIR="reviewer@ios.inf.uni-osnabrueck.de:/home/reviewer/generated"
SSH_HOST="reviewer@ios.inf.uni-osnabrueck.de"
SSH_PORT="2293"

# Use scp to copy files
scp -P $SSH_PORT -r "$SOURCE_DIR" "$TARGET_DIR"

# Check if scp was successful
if [ $? -eq 0 ]; then
  echo "Files copied successfully. Establishing SSH connection..."
  # SSH into the remote server
  ssh -p $SSH_PORT $SSH_HOST
else
  echo "Copy failed. SSH connection not established."
fi
