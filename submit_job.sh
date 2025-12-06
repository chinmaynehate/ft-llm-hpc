#!/bin/bash
#SBATCH --job-name=ft-llm-demo
#SBATCH --account=bfgy-delta-gpu                # <-- Your allocation
#SBATCH --partition=gpuA40x4               # <-- Adjust for your cluster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=4                         # <-- 4 ranks, one per GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ft-llm-%j.out
#SBATCH --error=ft-llm-%j.err

# Load modules
module load python/3.10
# module load cuda/12.4  # if needed

# Activate conda env if you have one
# source activate ft-env

# Environment variables
export WS_BASE="wss://ft-llm-relay-production.up.railway.app"
export WS_ROOM="ft-llm"
export NUM_GPUS=4
export NODE_ID=$(hostname)

# Model settings
export LLAMA_DEVICE="cuda:0"
export LLAMA_MAX_TOKENS=1024

# HuggingFace token (set this or use huggingface-cli login)
# export HF_TOKEN=hf_xxx

echo "========================================"
echo "FT-LLM Demo Job Starting (Phase 3)"
echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $NODE_ID"
echo "Num Tasks:    $SLURM_NTASKS"
echo "GPUs:         $NUM_GPUS"
echo "WebSocket:    $WS_BASE"
echo "========================================"

cd "$SLURM_SUBMIT_DIR"

# IMPORTANT: use srun to spawn 4 ranks
srun python hpc_main.py \
    --num-gpus $NUM_GPUS \
    --ws-base "$WS_BASE" \
    --room "$WS_ROOM" \
    --llama-device "$LLAMA_DEVICE" \
    --llama-max-new-tokens "$LLAMA_MAX_TOKENS"
