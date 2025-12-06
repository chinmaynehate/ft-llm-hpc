#!/usr/bin/env python3
"""
Test script to run HPC client locally (on your laptop)
to verify Railway connection works before deploying to HPC.
"""

import os
import asyncio
from hpc_ws_client import run_hpc_client

# Set your Railway URL here!
os.environ['WS_BASE'] = 'wss://ft-llm-relay-production.up.railway.app'
os.environ['WS_ROOM'] = 'ft-llm'
os.environ['NUM_GPUS'] = '4'

if __name__ == "__main__":
    print("Testing HPC client locally...")
    print("Press Ctrl+C to stop")
    asyncio.run(run_hpc_client())
