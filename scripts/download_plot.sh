#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

MACHINE_DIR="${SCRIPT_DIR}/../generated/"

TARGET_DIR="." # modify to your liking
SSH_HOST="reviewer@ios.inf.uni-osnabrueck.de"
SSH_PORT="2293"

# # Use scp to copy files
scp -P $SSH_PORT -r "$TARGET_DIR" "$MACHINE_DIR" 