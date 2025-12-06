#!/usr/bin/env python3
"""
HPC Main Entry Point for FT-LLM Demo

Phase 0: Just WebSocket client with fake GPU data
Later phases will add: NCCL, TP, model loading, RS coding, etc.
"""

import os
import sys
import asyncio
import argparse

from hpc_ws_client import HPCWebSocketClient


async def main():
    parser = argparse.ArgumentParser(description='FT-LLM HPC Client')
    parser.add_argument('--num-gpus', type=int, default=4, 
                        help='Number of GPUs (default: 4)')
    parser.add_argument('--ws-base', type=str, 
                        default='wss://ft-llm-relay-production.up.railway.app',
                        help='WebSocket base URL')
    parser.add_argument('--room', type=str, default='ft-llm',
                        help='Room name (default: ft-llm)')
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['WS_BASE'] = args.ws_base
    os.environ['WS_ROOM'] = args.room
    os.environ['NUM_GPUS'] = str(args.num_gpus)
    
    print("=" * 60)
    print("FT-LLM HPC Client - Phase 0")
    print("=" * 60)
    print(f"  WebSocket Base: {args.ws_base}")
    print(f"  Room:           {args.room}")
    print(f"  Num GPUs:       {args.num_gpus}")
    print("=" * 60)
    
    # Create and run client
    client = HPCWebSocketClient(num_gpus=args.num_gpus)
    
    try:
        await client.run()
    except KeyboardInterrupt:
        print("\n[HPC] Shutting down...")
        client.stop()


if __name__ == "__main__":
    asyncio.run(main())
