#!/bin/bash
#SBATCH --job-name=ft-llm-demo
#SBATCH --account=YOUR_ALLOCATION    # <-- CHANGE THIS
#SBATCH --partition=gpuA40x4         # <-- CHANGE based on Delta partition names
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ft-llm-%j.out
#SBATCH --error=ft-llm-%j.err

# Load modules (adjust based on Delta)
module load python/3.10
# module load cuda/12.2  # Uncomment if needed

# Activate your conda environment if you have one
# source activate ft-llm

# Set environment variables
export WS_BASE="wss://ft-llm-relay-production.up.railway.app"  # <-- Your Railway URL
export WS_ROOM="ft-llm"
export NUM_GPUS=4
export NODE_ID=$(hostname)

# Print info
echo "========================================"
echo "FT-LLM Demo Job Starting"
echo "========================================"
echo "Node: $NODE_ID"
echo "GPUs: $NUM_GPUS"
echo "WebSocket: $WS_BASE"
echo "========================================"

# Run the HPC client
cd $SLURM_SUBMIT_DIR
python hpc_main.py --num-gpus $NUM_GPUS --ws-base "$WS_BASE" --room "$WS_ROOM"
