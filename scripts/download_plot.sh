#!/bin/bash

# Get the directory where the script is located

MACHINE_DIR="/home/reviewer/Cinnamon/plot/" # Modify if you have cloned your own version

TARGET_DIR="." # modify to your liking
SSH_HOST="reviewer@ios.inf.uni-osnabrueck.de"
SSH_PORT="2293"

SOURCE_DIR="reviewer@ios.inf.uni-osnabrueck.de:$MACHINE_DIR"

# # Use scp to copy files
scp -P $SSH_PORT -r "$SOURCE_DIR" "$TARGET_DIR" 
