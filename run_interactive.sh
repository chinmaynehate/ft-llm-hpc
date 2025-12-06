#!/bin/bash
# ============================================
# FT-LLM Demo - Interactive Run Script (Phase 3)
# ============================================
#
# HOW TO USE:
#   1. Get allocation with 4 tasks:
#      salloc --nodes=1 --partition=gpuA40x4 --gpus-per-node=4 \
#             --ntasks=4 --cpus-per-task=8 --time=01:00:00 --account=YOUR_ACCOUNT
#
#   2. Run this script:
#      bash run_interactive.sh
#
# ============================================

# Configuration
export WS_BASE="${WS_BASE:-wss://ft-llm-relay-production.up.railway.app}"
export WS_ROOM="${WS_ROOM:-ft-llm}"
export NUM_GPUS="${NUM_GPUS:-4}"
export NODE_ID=$(hostname)

# Model settings
export LLAMA_DEVICE="${LLAMA_DEVICE:-cuda:0}"
export LLAMA_MAX_TOKENS="${LLAMA_MAX_TOKENS:-1024}"

echo "========================================"
echo "FT-LLM Demo - Phase 3 (NCCL + Real LLaMA)"
echo "========================================"
echo "Node:           $NODE_ID"
echo "GPUs (TP size): $NUM_GPUS"
echo "SLURM_NTASKS:   ${SLURM_NTASKS:-not set}"
echo "WebSocket:      $WS_BASE"
echo "LLaMA Device:   $LLAMA_DEVICE"
echo "Max Tokens:     $LLAMA_MAX_TOKENS"
echo "========================================"

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Check:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo "========================================"
fi

# Verify we have a multi-task allocation
if [ -z "$SLURM_NTASKS" ]; then
    echo "WARNING: SLURM_NTASKS not set!"
    echo "Make sure you allocated with --ntasks=4"
    echo ""
    echo "Example:"
    echo "  salloc --nodes=1 --partition=gpuA40x4 --gpus-per-node=4 \\"
    echo "         --ntasks=4 --cpus-per-task=8 --time=01:00:00 --account=YOUR_ACCOUNT"
    echo ""
    echo "Running in single-process mode for testing..."
    echo "========================================"
    
    # Single process fallback
    python hpc_main.py \
        --num-gpus 1 \
        --ws-base "$WS_BASE" \
        --room "$WS_ROOM" \
        --llama-device "$LLAMA_DEVICE" \
        --llama-max-new-tokens "$LLAMA_MAX_TOKENS"
else
    echo "Launching $SLURM_NTASKS ranks with srun..."
    echo "========================================"
    
    # Multi-process with srun
    srun python hpc_main.py \
        --num-gpus $NUM_GPUS \
        --ws-base "$WS_BASE" \
        --room "$WS_ROOM" \
        --llama-device "$LLAMA_DEVICE" \
        --llama-max-new-tokens "$LLAMA_MAX_TOKENS"
fi
