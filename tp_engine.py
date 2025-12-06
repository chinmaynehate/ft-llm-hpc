# tp_engine.py
"""
TPModelEngine: tensor-parallel aware abstraction around the LLaMA model.

Phase 4:
- Knows about (rank, world_size, local_rank).
- On rank 0, owns a real LlamaChatModel (unless fake_mode).
- On other ranks, it's a stub (no model loaded).
- Exposes a single async method handle_prompt(...) that:
    * For real mode: streams LLaMA output back via HPCWebSocketClient.
    * For fake mode or no GPU: falls back to client's default_prompt_handler.

Later:
- We can replace the internals with real tensor-parallel sharding (each rank
  holds a shard of the model, KV is distributed, etc.) without changing the
  WebSocket / UI wiring.
"""

from __future__ import annotations

from typing import Optional

import torch

from llama_model import LlamaChatModel
from hpc_ws_client import HPCWebSocketClient


class TPModelEngine:
    def __init__(
        self,
        rank: int,
        world_size: int,
        local_rank: int,
        fake_mode: bool,
        llama_device: str,
        max_new_tokens: int,
    ):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.fake_mode = fake_mode
        self.max_new_tokens = max_new_tokens

        self.llama: Optional[LlamaChatModel] = None

        # We *only* load the real model on rank 0 and when not in fake mode.
        # Other ranks will participate in NCCL heartbeats but won't load weights.
        if not fake_mode and torch.cuda.is_available() and rank == 0:
            print(
                f"[TPModelEngine] Initializing LLaMA on rank={rank}, "
                f"world_size={world_size}, device={llama_device}, "
                f"max_new_tokens={max_new_tokens}"
            )
            self.llama = LlamaChatModel(
                device=llama_device,
                max_new_tokens=max_new_tokens,
            )
        else:
            if rank == 0:
                reason = "fake_mode=True" if fake_mode else "CUDA not available"
                print(f"[TPModelEngine] Not loading model on rank 0: {reason}")
            else:
                print(f"[TPModelEngine] Rank {rank}: no model loaded (worker-only).")

    async def handle_prompt(
        self,
        client: HPCWebSocketClient,
        request_id: str,
        prompt: str,
    ) -> None:
        """
        Main entry point used by rank 0 to handle a UI prompt.

        Args:
            client: HPCWebSocketClient used to send tokens/events to UI.
            request_id: UI request identifier.
            prompt: User prompt text.
        """

        # Safety: we only expect rank 0 to handle prompts, but if a bug calls
        # this on another rank, just fall back to fake.
        if self.rank != 0:
            await client.send_event(
                f"⚠️ [rank {self.rank}] handle_prompt called unexpectedly; "
                f"falling back to fake response."
            )
            await client.default_prompt_handler(request_id, prompt)
            return

        # If we don't have a model (fake_mode or load failure), use the fake handler.
        if self.llama is None:
            await client.send_event(
                "🤖 LLaMA model not available on rank 0, using simulated response."
            )
            await client.default_prompt_handler(request_id, prompt)
            return

        # Real LLaMA path
        await client.send_event(
            f"🧠 [TP={self.world_size}, rank 0] Generating "
            f"(max {self.max_new_tokens} tokens)..."
        )

        try:
            token_count = 0
            async for chunk, finished in self.llama.stream_tokens(
                prompt,
                max_new_tokens=self.max_new_tokens,
            ):
                await client.send_token(request_id, chunk, finished=finished)
                token_count += len(chunk.split())

            await client.send_event(
                f"✅ [TP={self.world_size}] Generation complete (~{token_count} words)"
            )
        except Exception as e:
            err_msg = f"LLaMA error on rank 0: {e}"
            print("[TPModelEngine] " + err_msg)
            await client.send_event(f"❌ {err_msg}")
            await client.send_token(request_id, "\n[Model error]\n", finished=True)

