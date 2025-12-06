import os
import json
import asyncio
import socket
import uuid
import time
import random
import websockets


def build_ws_url(default_role="hpc"):
    """Build WebSocket URL from environment variables"""
    url = os.environ.get("WS_SERVER", "").strip()
    if url:
        return url
    
    base = os.environ.get("WS_BASE", "wss://ft-llm-relay-production.up.railway.app")
    room = os.environ.get("WS_ROOM", "ft-llm")
    cid = os.environ.get(
        "WS_CLIENT_ID",
        f"{default_role}-{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    )
    return f"{base}/ws/{room}/{cid}?role={default_role}"


class HPCWebSocketClient:
    def __init__(self, num_gpus: int = 4, auto_hotswap: bool = True, hotswap_delay: float = 5.0):
        self.ws_url = build_ws_url("hpc")
        self.num_gpus = num_gpus
        self.node_id = os.environ.get("NODE_ID", socket.gethostname())
        self.ws = None
        self.running = True
        
        # Hot-swap configuration
        self.auto_hotswap = auto_hotswap  # If True, GPU comes back after recovery
        self.hotswap_delay = hotswap_delay  # Seconds to wait before hot-swap
        
        # GPU states: 'healthy', 'failed', 'recovering'
        self.gpu_states = {i: 'healthy' for i in range(num_gpus)}
        
        # Simulated metrics
        self.base_throughput = 120.0  # tokens/sec at full TP
        self.base_latency = 25.0      # ms at full TP
        self.current_throughput = self.base_throughput
        self.current_latency = self.base_latency
        
        # Recovery simulation state
        self.recovery_in_progress = False
        self.recovery_start_time = None
        
        # Callbacks (for later phases)
        self.on_kill_gpu = None
        self.on_prompt = None
    
    @property
    def active_gpu_count(self):
        """Count of healthy GPUs"""
        return sum(1 for s in self.gpu_states.values() if s == 'healthy')
    
    def get_gpu_status(self):
        """Get current GPU status"""
        gpus = []
        for i in range(self.num_gpus):
            state = self.gpu_states.get(i, 'healthy')
            
            if state == 'failed':
                vram_used = 0.0
            elif state == 'recovering':
                vram_used = random.uniform(5.0, 15.0)  # Partial VRAM during recovery
            else:
                # Simulate VRAM usage (varies a bit)
                vram_used = random.uniform(15.0, 38.0)
            
            gpus.append({
                "id": i,
                "state": state,
                "vram_used_gb": vram_used,
                "vram_total_gb": 48.0,  # A40 has 48GB
            })
        
        return {
            "type": "gpu_status",
            "node_id": self.node_id,
            "gpus": gpus,
            "tp_world_size": self.active_gpu_count,
        }
    
    def get_metrics(self):
        """Get current performance metrics"""
        # Simulate throughput degradation based on active GPUs
        active = self.active_gpu_count
        total = self.num_gpus
        
        if active == 0:
            throughput = 0
            latency = 999
        else:
            # Throughput scales roughly linearly with GPU count
            # But not perfectly - there's some overhead
            efficiency = 0.9 + (active / total) * 0.1  # 90-100% efficiency
            throughput = (active / total) * self.base_throughput * efficiency
            
            # Latency increases when fewer GPUs (more work per GPU)
            latency = self.base_latency * (total / active)
            
            # Add some noise
            throughput += random.uniform(-5, 5)
            latency += random.uniform(-2, 2)
            
            # During recovery, add extra latency
            if self.recovery_in_progress:
                latency += 20
                throughput *= 0.7
        
        self.current_throughput = max(0, throughput)
        self.current_latency = max(0, latency)
        
        return {
            "type": "metrics",
            "throughput": round(self.current_throughput, 1),
            "latency": round(self.current_latency, 1),
            "tp_size": active,
            "active_gpus": active,
            "total_gpus": total,
        }
    
    async def send_event(self, msg: str):
        """Send an event message to UI"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "event",
                "msg": msg,
            }))
    
    async def send_token(self, request_id: str, token: str, finished: bool = False):
        """Send a token update to UI"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "token_update",
                "request_id": request_id,
                "token": token,
                "finished": finished,
            }))
    
    async def send_recovery_event(self, event: str, failed_rank: int, msg: str):
        """Send a recovery event to UI"""
        if self.ws:
            await self.ws.send(json.dumps({
                "type": "recovery_event",
                "event": event,
                "failed_rank": failed_rank,
                "msg": msg,
            }))
    
    async def simulate_recovery(self, gpu_id: int):
        """
        Simulate the full recovery process after GPU failure:
        1. Detect failure
        2. Abort NCCL
        3. Recreate NCCL with N-1 GPUs
        4. Recover KV cache from RS parity
        5. (Optional) Hot-swap: bring GPU back online
        """
        self.recovery_in_progress = True
        self.recovery_start_time = time.time()
        
        # ========================================
        # Phase 1: Degraded Mode Recovery (TP=4 → TP=3)
        # ========================================
        
        # Step 1: Detect failure
        await self.send_recovery_event("started", gpu_id, f"🚨 GPU {gpu_id} failure detected! Starting recovery...")
        await asyncio.sleep(0.5)
        
        # Step 2: NCCL abort
        await self.send_recovery_event("nccl_abort", gpu_id, "Aborting NCCL communicator...")
        await asyncio.sleep(0.3)
        
        # Step 3: NCCL recreate with surviving GPUs
        await self.send_recovery_event("nccl_recreate", gpu_id, f"Recreating NCCL communicator with TP={self.active_gpu_count}")
        await asyncio.sleep(0.5)
        
        # Step 4: KV cache recovery
        await self.send_recovery_event("kv_recovery", gpu_id, "Recovering KV cache from Reed-Solomon parity...")
        await asyncio.sleep(1.0)
        
        # Step 5: Degraded mode complete
        degraded_time = time.time() - self.recovery_start_time
        await self.send_recovery_event("degraded_complete", gpu_id, 
            f"✅ Degraded recovery complete in {degraded_time:.2f}s. Running at TP={self.active_gpu_count}")
        
        self.recovery_in_progress = False
        
        # ========================================
        # Phase 2: Hot-Swap (TP=3 → TP=4) - Optional
        # ========================================
        
        if self.auto_hotswap:
            # Wait before hot-swap (simulates time to restart/replace GPU)
            await asyncio.sleep(self.hotswap_delay)
            
            # Start hot-swap process
            await self.send_recovery_event("hotswap_started", gpu_id, 
                f"🔄 Hot-swap initiated: Bringing GPU {gpu_id} back online...")
            
            # Mark GPU as recovering (yellow state)
            self.gpu_states[gpu_id] = 'recovering'
            await asyncio.sleep(0.5)
            
            # Step 1: Load weights onto new GPU
            await self.send_recovery_event("hotswap_weights", gpu_id, 
                f"Loading model weights onto GPU {gpu_id}...")
            await asyncio.sleep(1.0)
            
            # Step 2: Abort current NCCL (again)
            await self.send_recovery_event("hotswap_nccl_abort", gpu_id, 
                "Aborting NCCL communicator for expansion...")
            await asyncio.sleep(0.3)
            
            # Step 3: Recreate NCCL with all GPUs
            self.gpu_states[gpu_id] = 'healthy'  # ← GPU turns GREEN here!
            await self.send_recovery_event("hotswap_nccl_recreate", gpu_id, 
                f"Recreating NCCL communicator with TP={self.active_gpu_count}")
            await asyncio.sleep(0.5)
            
            # Step 4: Redistribute KV cache
            await self.send_recovery_event("hotswap_kv_redistribute", gpu_id, 
                "Redistributing KV cache shards...")
            await asyncio.sleep(0.5)
            
            # Step 5: Hot-swap complete!
            total_time = time.time() - self.recovery_start_time
            await self.send_recovery_event("complete", gpu_id, 
                f"🎉 Hot-swap complete! Back to TP={self.active_gpu_count}. Total recovery time: {total_time:.2f}s")
    
    async def handle_message(self, data: dict):
        """Handle incoming message from UI"""
        mtype = data.get("type")
        
        if mtype == "kill_gpu":
            gpu_id = data.get("gpu_id")
            print(f"[HPC] Received kill_gpu for GPU {gpu_id}")
            
            if self.gpu_states.get(gpu_id) == 'healthy':
                # Mark as failed
                self.gpu_states[gpu_id] = 'failed'
                await self.send_event(f"💀 GPU {gpu_id} killed!")
                
                # Start recovery simulation in background
                asyncio.create_task(self.simulate_recovery(gpu_id))
            else:
                await self.send_event(f"GPU {gpu_id} is already {self.gpu_states.get(gpu_id)}")
                
        elif mtype == "submit_prompt":
            request_id = data.get("request_id")
            prompt = data.get("prompt", "")
            print(f"[HPC] Received prompt: {prompt[:50]}...")
            
            # Call callback if set, otherwise use default fake response
            if self.on_prompt:
                await self.on_prompt(request_id, prompt)
            else:
                await self.default_prompt_handler(request_id, prompt)
    
    async def default_prompt_handler(self, request_id: str, prompt: str):
        """Default handler: generate fake tokens"""
        active = self.active_gpu_count
        
        if active == 0:
            await self.send_token(request_id, "ERROR: No active GPUs available!\n", finished=True)
            return
        
        # Simulate response based on current TP size
        response = f"[TP={active}] Processing: '{prompt[:40]}...'\n\n"
        response += f"Current throughput: {self.current_throughput:.1f} tok/s\n"
        response += f"Current latency: {self.current_latency:.1f} ms\n\n"
        response += "This is a simulated response. Real inference will come in later phases!\n"
        
        # Simulate streaming with delay based on throughput
        words = response.split(' ')
        for i, word in enumerate(words):
            token = word + (' ' if i < len(words) - 1 else '')
            await self.send_token(request_id, token, finished=False)
            
            # Delay based on current throughput
            delay = 1.0 / max(self.current_throughput / 10, 1)  # Rough simulation
            await asyncio.sleep(min(delay, 0.2))
        
        await self.send_token(request_id, "", finished=True)
    
    async def status_loop(self):
        """Send GPU status periodically"""
        while self.running:
            if self.ws:
                try:
                    status = self.get_gpu_status()
                    await self.ws.send(json.dumps(status))
                except Exception as e:
                    print(f"[HPC] Error sending status: {e}")
            await asyncio.sleep(1.0)
    
    async def metrics_loop(self):
        """Send metrics periodically"""
        while self.running:
            if self.ws:
                try:
                    metrics = self.get_metrics()
                    await self.ws.send(json.dumps(metrics))
                except Exception as e:
                    print(f"[HPC] Error sending metrics: {e}")
            await asyncio.sleep(1.0)  # Send metrics every second
    
    async def receive_loop(self):
        """Receive and handle messages"""
        while self.running:
            try:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=0.1)
                data = json.loads(msg)
                await self.handle_message(data)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                print("[HPC] Connection closed")
                break
            except Exception as e:
                print(f"[HPC] Error receiving: {e}")
                continue
    
    async def run(self):
        """Main run loop with reconnection"""
        while self.running:
            try:
                print(f"[HPC] Connecting to: {self.ws_url}")
                
                async with websockets.connect(
                    self.ws_url,
                    max_size=10 * 1024 * 1024,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    self.ws = ws
                    print("[HPC] Connected!")
                    
                    # Send initial event
                    await self.send_event(f"🚀 Node {self.node_id} online with {self.num_gpus} GPUs (TP={self.num_gpus})")
                    
                    # Run all loops concurrently
                    await asyncio.gather(
                        self.status_loop(),
                        self.metrics_loop(),
                        self.receive_loop(),
                    )
                    
            except Exception as e:
                print(f"[HPC] Connection error: {e}")
                print("[HPC] Reconnecting in 3 seconds...")
                self.ws = None
                await asyncio.sleep(3)
        
        print("[HPC] Client stopped")
    
    def stop(self):
        """Stop the client"""
        self.running = False


async def run_hpc_client():
    """Convenience function to run the client"""
    num_gpus = int(os.environ.get("NUM_GPUS", "4"))
    auto_hotswap = os.environ.get("AUTO_HOTSWAP", "1") == "1"
    hotswap_delay = float(os.environ.get("HOTSWAP_DELAY", "5.0"))
    
    client = HPCWebSocketClient(
        num_gpus=num_gpus,
        auto_hotswap=auto_hotswap,
        hotswap_delay=hotswap_delay
    )
    await client.run()


if __name__ == "__main__":
    asyncio.run(run_hpc_client())
