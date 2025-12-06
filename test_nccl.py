#!/usr/bin/env python3
"""
Test NCCL initialization under Slurm.

Run with:
    srun --ntasks=4 python test_nccl.py
"""

import os
import torch
import torch.distributed as dist
from dist_init import init_distributed, destroy_distributed


def main():
    rank, world_size, local_rank = init_distributed()
    
    print(f"[Rank {rank}] Initialized: world_size={world_size}, local_rank={local_rank}")
    
    if not dist.is_initialized():
        print(f"[Rank {rank}] Not distributed, exiting.")
        return
    
    # Simple all-reduce test
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([float(rank)], device=device)
    
    print(f"[Rank {rank}] Before all-reduce: {tensor.item()}")
    
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] After all-reduce: {tensor.item()}")
    # Expected: 0 + 1 + 2 + 3 = 6.0
    
    # Barrier to sync all ranks
    dist.barrier()
    
    if rank == 0:
        print("\n" + "=" * 40)
        print("✅ NCCL test passed!")
        print(f"   All {world_size} ranks successfully participated")
        print(f"   Sum = {tensor.item()} (expected: {sum(range(world_size))})")
        print("=" * 40)
    
    destroy_distributed()


if __name__ == "__main__":
    main()
