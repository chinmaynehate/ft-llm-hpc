#!/usr/bin/env python3
"""
Test LLaMA model locally (on your laptop or HPC)
without needing the WebSocket connection.
"""

import os
import asyncio
import torch

def main():
    print("=" * 60)
    print("LLaMA Model Local Test")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("   This test requires a GPU.")
        return

    print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
    print(f"   GPU 0: {torch.cuda.get_device_name(0)}")
    print()

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"✅ HF_TOKEN is set (length: {len(hf_token)})")
    else:
        print("⚠️  HF_TOKEN not set - may fail for gated models")
    print()

    # Load model
    print("Loading model...")
    from llama_model import LlamaChatModel

    try:
        model = LlamaChatModel(device="cuda:0", max_new_tokens=100)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print()
        print("Common issues:")
        print("  1. HF_TOKEN not set or invalid")
        print("  2. Haven't accepted LLaMA license on HuggingFace")
        print("  3. No internet connection to download model")
        return

    print()
    print("=" * 60)
    print("Testing generation...")
    print("=" * 60)

    async def test_streaming():
        prompt = "Explain machine learning in two sentences."
        print(f"Prompt: {prompt}")
        print()
        print("Response: ", end="", flush=True)

        async for chunk, finished in model.stream_tokens(prompt, chunk_size=2):
            print(chunk, end="", flush=True)

        print()
        print()
        print("=" * 60)
        print("✅ Test complete!")
        print("=" * 60)

    asyncio.run(test_streaming())


if __name__ == "__main__":
    main()
