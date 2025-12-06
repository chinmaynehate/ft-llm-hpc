#!/usr/bin/env python3
"""
HPC Main Entry Point for FT-LLM Demo - Phase 3

Now runs with torch.distributed (NCCL) if started with multiple tasks.

- Rank 0:
    * Loads LLaMA model (unless --fake-mode)
    * Creates WebSocket client and serves UI requests
- Ranks 1..world_size-1:
    * Join NCCL group
    * Run a simple all-reduce loop as a "heartbeat"
"""

import os
import asyncio
import argparse
import torch
import torch.distributed as dist

from hpc_ws_client import HPCWebSocketClient
from llama_model import LlamaChatModel
from dist_init import init_distributed, destroy_distributed


async def distributed_worker_loop(rank: int, world_size: int) -> None:
    """
    Simple NCCL heartbeat loop for non-zero ranks.
    
    Each worker maintains a tensor with its rank id and participates
    in an all-reduce every second. Rank 0 occasionally logs the sum.
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

            # Log occasionally (only from this rank to avoid spam)
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

    print(f"[Rank 0] Starting NCCL heartbeat (background)")

    try:
        while True:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            step += 1

            if step % 30 == 0:
                # Sum of ranks 0+1+2+3 = 6
                print(f"[Rank 0] NCCL heartbeat step={step}, sum={tensor.item():.1f}")

            tensor.fill_(0.0)
            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        print("[Rank 0] NCCL heartbeat cancelled.")
    except Exception as e:
        print(f"[Rank 0] NCCL heartbeat error: {e}")


async def main():
    parser = argparse.ArgumentParser(description='FT-LLM HPC Client - Phase 3')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs (default: 4)')
    parser.add_argument('--ws-base', type=str,
                        default='wss://ft-llm-relay-production.up.railway.app',
                        help='WebSocket base URL')
    parser.add_argument('--room', type=str, default='ft-llm',
                        help='Room name (default: ft-llm)')
    parser.add_argument('--llama-device', type=str, default='cuda:0',
                        help='Device for Llama model (only used on rank 0)')
    parser.add_argument('--llama-max-new-tokens', type=int, default=512,
                        help='Max new tokens per generation')
    parser.add_argument('--fake-mode', action='store_true',
                        help='Use fake inference instead of real model')
    args = parser.parse_args()

    # 1) Initialize torch.distributed FIRST (before any other setup)
    rank, world_size, local_rank = init_distributed(backend="nccl")

    # Only rank 0 prints the banner to avoid spam
    if rank == 0:
        print("=" * 60)
        print("FT-LLM HPC Client - Phase 3 (NCCL + Real LLaMA)")
        print("=" * 60)
        print(f"  WebSocket Base: {args.ws_base}")
        print(f"  Room:           {args.room}")
        print(f"  Requested GPUs: {args.num_gpus}")
        print(f"  LLaMA device:   {args.llama_device}")
        print(f"  Max tokens:     {args.llama_max_new_tokens}")
        print(f"  Fake mode:      {args.fake_mode}")
        print(f"  World size:     {world_size}")
        print("=" * 60)

    print(f"[HPC] rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # WebSocket configuration env vars
    os.environ['WS_BASE'] = args.ws_base
    os.environ['WS_ROOM'] = args.room
    os.environ['NUM_GPUS'] = str(world_size if world_size > 1 else args.num_gpus)

    # Effective number of GPUs = distributed world size
    effective_num_gpus = world_size if world_size > 1 else args.num_gpus

    # 2) Branch on rank
    if rank != 0:
        # ============================
        # Non-zero ranks: worker loop
        # ============================
        try:
            await distributed_worker_loop(rank, world_size)
        finally:
            destroy_distributed()
        return

    # ============================
    # Rank 0: Main server
    # ============================

    # Load model (optional fake mode)
    llama = None
    if not args.fake_mode:
        try:
            if torch.cuda.is_available():
                # Use local_rank for device to ensure rank 0 uses GPU 0
                device = f"cuda:{local_rank}"
                llama = LlamaChatModel(
                    device=device,
                    max_new_tokens=args.llama_max_new_tokens,
                )
            else:
                print("[HPC] CUDA not available, falling back to fake mode")
        except Exception as e:
            print(f"[HPC] Failed to load model: {e}")
            print("[HPC] Falling back to fake mode")
            llama = None

    # Create WebSocket client with num_gpus = world_size
    client = HPCWebSocketClient(num_gpus=effective_num_gpus)

    # Prompt handler
    async def on_prompt(request_id: str, prompt: str):
        if llama is None:
            await client.default_prompt_handler(request_id, prompt)
            return

        await client.send_event(
            f"🧠 [rank 0, TP={effective_num_gpus}] Generating (max {args.llama_max_new_tokens} tokens)..."
        )

        try:
            token_count = 0
            async for chunk, finished in llama.stream_tokens(
                prompt,
                max_new_tokens=args.llama_max_new_tokens,
            ):
                await client.send_token(request_id, chunk, finished=finished)
                token_count += len(chunk.split())

            await client.send_event(f"✅ Generation complete (~{token_count} words)")

        except Exception as e:
            err_msg = f"Model error: {e}"
            print("[HPC] " + err_msg)
            await client.send_event(f"❌ {err_msg}")
            await client.send_token(request_id, "\n[Model error]\n", finished=True)

    client.on_prompt = on_prompt

    # 3) Run WebSocket client + NCCL heartbeat concurrently
    try:
        if world_size > 1:
            # Run both WebSocket client and NCCL heartbeat
            await asyncio.gather(
                client.run(),
                rank0_worker_loop(world_size),
            )
        else:
            # Single process mode - just run WebSocket client
            await client.run()

    except KeyboardInterrupt:
        print("\n[HPC] Shutting down (KeyboardInterrupt)...")
        client.stop()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    asyncio.run(main())
