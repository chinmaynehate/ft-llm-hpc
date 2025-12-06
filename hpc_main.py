#!/usr/bin/env python3
"""
HPC Main Entry Point for FT-LLM Demo - Phase 2
"""

import os
import asyncio
import argparse

from hpc_ws_client import HPCWebSocketClient
from llama_model import LlamaChatModel


async def main():
    parser = argparse.ArgumentParser(description='FT-LLM HPC Client')
    parser.add_argument('--num-gpus', type=int, default=4,
                        help='Number of GPUs (default: 4)')
    parser.add_argument('--ws-base', type=str,
                        default='wss://ft-llm-relay-production.up.railway.app',
                        help='WebSocket base URL')
    parser.add_argument('--room', type=str, default='ft-llm',
                        help='Room name (default: ft-llm)')
    parser.add_argument('--llama-device', type=str, default='cuda:0',
                        help='Device for Llama model')
    parser.add_argument('--llama-max-new-tokens', type=int, default=512,
                        help='Max new tokens per generation (default: 512 for demo)')
    parser.add_argument('--fake-mode', action='store_true',
                        help='Use fake inference instead of real model')
    args = parser.parse_args()

    os.environ['WS_BASE'] = args.ws_base
    os.environ['WS_ROOM'] = args.room
    os.environ['NUM_GPUS'] = str(args.num_gpus)

    print("=" * 60)
    print("FT-LLM HPC Client - Phase 2 (Real Llama)")
    print("=" * 60)
    print(f"  WebSocket Base: {args.ws_base}")
    print(f"  Room:           {args.room}")
    print(f"  Num GPUs:       {args.num_gpus}")
    print(f"  Llama device:   {args.llama_device}")
    print(f"  Max tokens:     {args.llama_max_new_tokens}")
    print(f"  Fake mode:      {args.fake_mode}")
    print("=" * 60)

    # Load model
    llama = None
    if not args.fake_mode:
        try:
            import torch
            if torch.cuda.is_available():
                llama = LlamaChatModel(
                    device=args.llama_device,
                    max_new_tokens=args.llama_max_new_tokens,
                )
            else:
                print("[HPC] CUDA not available, falling back to fake mode")
        except Exception as e:
            print(f"[HPC] Failed to load model: {e}")
            print("[HPC] Falling back to fake mode")

    # Create WebSocket client
    client = HPCWebSocketClient(num_gpus=args.num_gpus)

    # Prompt handler
    async def on_prompt(request_id: str, prompt: str):
        if llama is None:
            await client.default_prompt_handler(request_id, prompt)
            return

        await client.send_event(f"🧠 Generating response (max {args.llama_max_new_tokens} tokens)...")

        try:
            token_count = 0
            async for chunk, finished in llama.stream_tokens(prompt):
                await client.send_token(request_id, chunk, finished=finished)
                token_count += len(chunk.split())
            
            await client.send_event(f"✅ Generation complete (~{token_count} words)")
        except Exception as e:
            err_msg = f"Model error: {e}"
            print("[HPC] " + err_msg)
            await client.send_event(f"❌ {err_msg}")
            await client.send_token(request_id, "\n[Model error]\n", finished=True)

    client.on_prompt = on_prompt

    try:
        await client.run()
    except KeyboardInterrupt:
        print("\n[HPC] Shutting down...")
        client.stop()


if __name__ == "__main__":
    asyncio.run(main())
