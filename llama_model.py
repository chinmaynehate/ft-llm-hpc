# llama_model.py
"""
LlamaChatModel: wraps a real Llama-3.2-1B-Instruct model from Hugging Face.

Phase 2: single-GPU, single-process inference.
Later phases: we'll replace this with TP-aware engine.
"""

import os
from typing import AsyncGenerator, Tuple, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL_ID = os.environ.get(
    "LLAMA_MODEL_ID",
    "meta-llama/Llama-3.2-1B-Instruct",
)


class LlamaChatModel:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda:0",
        max_new_tokens: int = 512,      # Increased default for demo
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        self.model_id = model_id
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        hf_token = os.environ.get("HF_TOKEN")

        print(f"[Llama] Loading model: {model_id} on device={device}")
        print(f"[Llama] Max new tokens: {max_new_tokens}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
            token=hf_token,
        )

        self.model.to(self.device)
        self.model.eval()

        # System prompt that encourages LONG, DETAILED responses
        self.system_prompt = (
            "You are a helpful, knowledgeable assistant. "
            "When asked a question, provide a comprehensive, detailed explanation. "
            "Include examples, context, and thorough coverage of the topic. "
            "Aim for detailed, educational responses."
        )

        print("[Llama] Model loaded and ready.")

    def _build_inputs(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Build input IDs and attention mask using the chat template."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            return_attention_mask=True,
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    @torch.inference_mode()
    def generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate a full completion for the given prompt.
        
        Args:
            prompt: The user's input prompt
            max_new_tokens: Override default max tokens (optional)
        """
        inputs = self._build_inputs(prompt)
        input_length = inputs["input_ids"].shape[1]
        
        # Use provided max_new_tokens or default
        tokens_to_generate = max_new_tokens or self.max_new_tokens

        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=tokens_to_generate,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            # Prevent early stopping for longer outputs
            min_new_tokens=min(100, tokens_to_generate),  # At least 100 tokens
        )

        generated = output_ids[0, input_length:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()

    async def stream_tokens(
        self,
        prompt: str,
        max_new_tokens: int = None,
        chunk_size: int = 2,           # Smaller chunks = more "live" feeling
        min_delay: float = 0.03,       # Slightly slower for demo visibility
        max_delay: float = 0.10,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """
        Async generator that yields (token_chunk, finished) pairs.
        """
        import asyncio
        import random

        text = self.generate_text(prompt, max_new_tokens=max_new_tokens)

        if not text:
            yield "No output from model.", True
            return

        words = text.split(" ")
        buffer = []

        for i, w in enumerate(words):
            buffer.append(w)

            if len(buffer) >= chunk_size or i == len(words) - 1:
                chunk = " ".join(buffer) + (" " if i < len(words) - 1 else "")
                buffer = []
                finished = (i == len(words) - 1)
                yield chunk, finished

                delay = random.uniform(min_delay, max_delay)
                await asyncio.sleep(delay)


if __name__ == "__main__":
    import asyncio

    async def test():
        print("Testing LlamaChatModel with long output...")

        if not torch.cuda.is_available():
            print("CUDA not available!")
            return

        # Test with 512 tokens
        model = LlamaChatModel(device="cuda:0", max_new_tokens=512)

        prompt = "Explain the history and evolution of artificial intelligence from the 1950s to today."
        print(f"\nPrompt: {prompt}")
        print("\nResponse: ", end="", flush=True)

        word_count = 0
        async for chunk, finished in model.stream_tokens(prompt):
            print(chunk, end="", flush=True)
            word_count += len(chunk.split())

        print(f"\n\n[Generated ~{word_count} words]")
        print("Test complete!")

    asyncio.run(test())
