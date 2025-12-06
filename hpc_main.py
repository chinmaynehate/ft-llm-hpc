#!/usr/bin/env python3
"""
HPC Main Entry Point for FT-LLM Demo - Phase 4

Now runs with torch.distributed (NCCL) if started with multiple tasks.

- Rank 0:
    * Owns TPModelEngine (which in turn may own a real LLaMA model)
    * Creates WebSocket client and serves UI requests
    * Participates in NCCL heartbeat in the background

- Ranks 1..world_size-1:
    * Join NCCL group
    * Run a simple all-reduce loop as a "heartbeat"

The TPModelEngine is the abstraction we will later extend for real
multi-GPU tensor parallelism and KV/RS distribution.
"""

import os
import asyncio
import argparse

import torch
import torch.distributed as dist

from hpc_ws_client import HPCWebSocketClient
from dist_init import init_distributed, destroy_distributed
from tp_engine import TPModelEngine


async def distributed_worker_loop(rank: int, world_size: int) -> None:
    """
    Simple NCCL heartbeat loop for non-zero ranks.

    Each worker maintains a tensor with its rank id and participates
    in an all-reduce every second. Logs occasionally so we can see NCCL is alive.
    """
    if not dist.is_initialized() or world_size <= 1:
        print(f"[Worker {rank}] dist not initialized (world_size={world_size}), idle.")
        while True:
            await asyncio.sleep(5.0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([float(rank)], device=device)
    step = 0

    print(f"[Worker {rank}] starting NCCL heartbeat loop on device {device}")

    try:
        while True:
            # All-reduce sum across all ranks
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            step += 1

            if step % 30 == 0:
                print(f"[Worker {rank}] NCCL heartbeat step={step}, sum={tensor.item():.1f}")

            # Reset tensor for next iteration
            tensor.fill_(float(rank))
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        print(f"[Worker {rank}] cancelled, exiting loop.")
    except Exception as e:
        print(f"[Worker {rank}] error in NCCL loop: {e}")


async def rank0_worker_loop(world_size: int) -> None:
    """
    NCCL heartbeat loop for rank 0 (runs alongside WebSocket client).

    This keeps rank 0 participating in the all-reduce operations
    initiated by other ranks.
    """
    if not dist.is_initialized() or world_size <= 1:
        return

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tensor = torch.tensor([0.0], device=device)  # Rank 0
    step = 0

    print(f"[Rank 0] Starting NCCL heartbeat (background) on {device}")

    try:
        while True:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            step += 1

            if step % 30 == 0:
                print(f"[Rank 0] NCCL heartbeat step={step}, sum={tensor.item():.1f}")

            tensor.fill_(0.0)
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        print("[Rank 0] NCCL heartbeat cancelled.")
    except Exception as e:
        print(f"[Rank 0] NCCL heartbeat error: {e}")


async def main():
    parser = argparse.ArgumentParser(description='FT-LLM HPC Client - Phase 4')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs (default: 4). '
                             'Used as a hint; actual TP size = world_size.')
    parser.add_argument('--ws-base', type=str,
                        default='wss://ft-llm-relay-production.up.railway.app',
                        help='WebSocket base URL')
    parser.add_argument('--room', type=str, default='ft-llm',
                        help='Room name (default: ft-llm)')
    parser.add_argument('--llama-device', type=str, default='cuda:0',
                        help='Device string for LLaMA when running single-process')
    parser.add_argument('--llama-max-new-tokens', type=int, default=512,
                        help='Max new tokens per generation')
    parser.add_argument('--fake-mode', action='store_true',
                        help='Use fake inference instead of real model')
    args = parser.parse_args()

    # 1) Initialize torch.distributed FIRST
    rank, world_size, local_rank = init_distributed(backend="nccl")

    # Only rank 0 prints the big banner to avoid spam
    if rank == 0:
        print("=" * 60)
        print("FT-LLM HPC Client - Phase 4 (TP Engine + NCCL + Real LLaMA)")
        print("=" * 60)
        print(f"  WebSocket Base: {args.ws_base}")
        print(f"  Room:           {args.room}")
        print(f"  Requested GPUs: {args.num_gpus}")
        print(f"  World size:     {world_size}")
        print(f"  LLaMA device:   {args.llama_device}")
        print(f"  Max tokens:     {args.llama_max_new_tokens}")
        print(f"  Fake mode:      {args.fake_mode}")
        print("=" * 60)

    print(f"[HPC] rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # WebSocket configuration env vars (used by HPCWebSocketClient)
    os.environ['WS_BASE'] = args.ws_base
    os.environ['WS_ROOM'] = args.room
    os.environ['NUM_GPUS'] = str(world_size if world_size > 1 else args.num_gpus)

    # Effective TP size = distributed world size if >1, else num_gpus hint
    effective_num_gpus = world_size if world_size > 1 else args.num_gpus

    # ==============================================================
    # Non-zero ranks: just run the worker heartbeat loop
    # ==============================================================
    if rank != 0:
        try:
            await distributed_worker_loop(rank, world_size)
        finally:
            destroy_distributed()
        return

    # ==============================================================
    # Rank 0: main TP engine + WebSocket server
    # ==============================================================

    # Choose device for LLaMA on rank 0:
    # - In multi-process mode, prefer cuda:local_rank
    # - In single-process mode, use the CLI arg directly
    if world_size > 1 and torch.cuda.is_available():
        llama_device = f"cuda:{local_rank}"
    else:
        llama_device = args.llama_device

    # Create the TP engine (may or may not load a real model based on fake_mode)
    tp_engine = TPModelEngine(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        fake_mode=args.fake_mode,
        llama_device=llama_device,
        max_new_tokens=args.llama_max_new_tokens,
    )

    # Create WebSocket client with num_gpus = TP size
    client = HPCWebSocketClient(num_gpus=effective_num_gpus)

    # Prompt handler wired into TP engine
    async def on_prompt(request_id: str, prompt: str):
        await tp_engine.handle_prompt(client, request_id, prompt)

    client.on_prompt = on_prompt

    # 3) Run WebSocket client + NCCL heartbeat concurrently
    try:
        if world_size > 1:
            await asyncio.gather(
                client.run(),          # handles UI <-> HPC traffic
                rank0_worker_loop(world_size),  # keeps rank 0 in NCCL collectives
            )
        else:
            # Single-process mode: just run client
            await client.run()
    except KeyboardInterrupt:
        print("\n[HPC] Shutting down (KeyboardInterrupt)...")
        client.stop()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    asyncio.run(main())

