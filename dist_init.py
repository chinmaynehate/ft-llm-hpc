# dist_init.py
"""
Utilities for initializing torch.distributed (NCCL) under Slurm or locally.
Phase 3: basic single-node multi-process setup.
"""

import os
from typing import Tuple
import torch
import torch.distributed as dist


def init_distributed(backend: str = "nccl") -> Tuple[int, int, int]:
    """
    Initialize torch.distributed if running with world_size > 1.
    
    Returns:
        (rank, world_size, local_rank)
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank

    # Detect Slurm
    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        # Rank within the node (0..gpus_per_node-1)
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        # Fallback for local / single-process
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        # For single-node jobs, localhost works fine
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        print(f"[dist] initialized backend={backend}, rank={rank}, "
              f"world_size={world_size}, local_rank={local_rank}")
    else:
        print("[dist] running in single-process mode (world_size=1)")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def destroy_distributed():
    """Tear down the process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# Test distributed setup
if __name__ == "__main__":
    rank, world_size, local_rank = init_distributed()
    print(f"[Test] rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    if dist.is_initialized():
        # Simple all-reduce test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor([float(rank)], device=device)
        print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"[Rank {rank}] After all-reduce (sum): {tensor.item()}")
        
        # Expected sum = 0 + 1 + 2 + 3 = 6 for world_size=4
        
    destroy_distributed()
    print(f"[Rank {rank}] Done.")
