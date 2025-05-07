#!/bin/bash

# Get the directory where the script is located
# SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# SOURCE_DIR="${SCRIPT_DIR}/../cinnamon/testbench/generated/"

# TARGET_DIR="reviewer@ios.inf.uni-osnabrueck.de:/home/reviewer/generated"
# SSH_HOST="reviewer@ios.inf.uni-osnabrueck.de"
# SSH_PORT="2293"

# # Use scp to copy files
# scp -P $SSH_PORT -r "$SOURCE_DIR" "$TARGET_DIR"

# # Check if scp was successful
# if [ $? -eq 0 ]; then
#   echo "Files copied successfully. Establishing SSH connection..."
#   # SSH into the remote server
#   ssh -p $SSH_PORT $SSH_HOST
# else
#   echo "Copy failed. SSH connection not established."
# fi

cd /home/reviewer/Cinnamon/cinnamon/testbench/ || { echo "Failed to change directory to /home/reviewer/generated"; exit 1; }
source /opt/upmem/upmem-2023.2.0-Linux-x86_64/upmem_env.sh 
python3 get_results.py

cd ../../

cat ./cinnamon/testbench/exp-fig-11.txt
cat ./cinnamon/testbench/exp-fig-12.txt
cp ./cinnamon/testbench/exp-fig-11.txt ./plot/exp-fig-11.txt
cp ./cinnamon/testbench/exp-fig-12.txt ./plot/exp-fig-12.txt
