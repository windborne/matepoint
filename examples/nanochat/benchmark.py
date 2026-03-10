"""
Matepoint activation offloading benchmark with nanochat.

Demonstrates matepoint's VRAM savings on Karpathy's nanochat GPT models.
Compares baseline (no offloading) vs matepoint across batch sizes.

Setup:
    git clone https://github.com/karpathy/nanochat.git
    cd nanochat
    uv sync --extra gpu
    uv pip install matepoint

    # Download minimal data and train tokenizer
    export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
    python -m nanochat.dataset -n 8
    python -m scripts.tok_train

Usage (run from nanochat repo root):
    # Compare baseline vs matepoint at a specific config
    python path/to/benchmark.py --depth=12 --batch-size=8

    # Find OOM boundary: try baseline, then matepoint at 2x batch
    python path/to/benchmark.py --depth=20 --batch-size=4 --mode=baseline
    python path/to/benchmark.py --depth=20 --batch-size=8 --mode=matepoint

    # Full sweep across batch sizes
    python path/to/benchmark.py --depth=12 --sweep

RTX 4090 (24 GB) results:

    d12 (286M params):
    | Batch | Baseline VRAM | Matepoint VRAM | Throughput overhead |
    |-------|---------------|----------------|---------------------|
    | 8     | 16.04 GB      | 10.63 GB (-34%)| 22%                 |
    | 16    | OOM           | 18.99 GB       | -                   |

    d20 (897M params):
    | Batch | Baseline VRAM | Matepoint VRAM | Throughput overhead |
    |-------|---------------|----------------|---------------------|
    | 4     | 20.03 GB      | 12.04 GB (-40%)| 23%                 |
    | 8     | OOM           | 16.48 GB       | -                   |
"""

import os
import sys
import gc
import time
import argparse
import json

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F

import matepoint


def build_model(depth, vocab_size, seq_len, device):
    """Build a nanochat GPT model on the given device."""
    from nanochat.gpt import GPT, GPTConfig

    aspect_ratio = 64
    head_dim = 128
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim

    config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",  # full context for SDPA compatibility
    )

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    return model


def patch_forward_with_matepoint(model):
    """Monkey-patch model.forward to wrap each transformer block with matepoint.checkpoint."""
    import types
    from nanochat.common import COMPUTE_DTYPE

    def matepoint_forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            # Each block's activations are offloaded to CPU RAM via matepoint
            x = matepoint.checkpoint(
                block, x, ve, cos_sin, self.window_sizes[i], None,
                use_reentrant=False,
            )
        x = F.rms_norm(x, (x.size(-1),))

        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits

    model.forward = types.MethodType(matepoint_forward, model)


def run_benchmark(depth, batch_size, seq_len, num_steps, use_matepoint, device_id=0):
    """Run a training benchmark. Returns a results dict."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    from nanochat.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    model = build_model(depth, vocab_size, seq_len, device)
    num_params = sum(p.numel() for p in model.parameters())
    mode_str = "matepoint" if use_matepoint else "baseline"

    print(f"\n{'='*60}")
    print(f"  {mode_str.upper()} | d{depth} ({num_params/1e6:.0f}M) | batch={batch_size}")
    print(f"{'='*60}")

    if use_matepoint:
        patch_forward_with_matepoint(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    step_times = []
    try:
        for step in range(num_steps):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

            t0 = time.time()
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t0

            if step >= 2:
                step_times.append(dt)

            peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            tps = batch_size * seq_len / dt
            if step < 3 or step % 5 == 0:
                print(f"  step {step:3d} | loss {loss.item():.4f} | {dt*1000:.0f}ms | "
                      f"{tps:,.0f} tok/s | peak {peak_gb:.2f} GB")

        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        avg_dt = sum(step_times) / len(step_times) if step_times else 0
        avg_tps = batch_size * seq_len / avg_dt if avg_dt > 0 else 0

        print(f"\n  Peak VRAM: {peak_gb:.2f} GB | Avg: {avg_dt*1000:.0f}ms/step, {avg_tps:,.0f} tok/s")
        return dict(mode=mode_str, depth=depth, batch_size=batch_size, params=num_params,
                    peak_vram_gb=round(peak_gb, 2), avg_step_ms=round(avg_dt*1000, 1),
                    avg_tok_per_s=round(avg_tps), status="ok")

    except torch.cuda.OutOfMemoryError:
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"\n  OOM at step {step}! (peak was {peak_gb:.2f} GB)")
        return dict(mode=mode_str, depth=depth, batch_size=batch_size, params=num_params,
                    peak_vram_gb=round(peak_gb, 2), status="OOM")


def run_sweep(depth, seq_len, num_steps, device_id):
    """Sweep batch sizes for both modes, finding max batch for each."""
    import subprocess

    batch_sizes = [4, 8, 12, 16, 24, 32]
    results = []

    for mode in ["baseline", "matepoint"]:
        for bs in batch_sizes:
            cmd = [sys.executable, __file__,
                   f"--depth={depth}", f"--batch-size={bs}", f"--seq-len={seq_len}",
                   f"--steps={num_steps}", f"--gpu={device_id}", f"--mode={mode}",
                   "--json"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            # Parse JSON from last line
            for line in reversed(proc.stdout.strip().split('\n')):
                try:
                    r = json.loads(line)
                    results.append(r)
                    if r["status"] == "OOM":
                        print(f"  {mode} batch={bs}: OOM — stopping sweep for this mode")
                    break
                except json.JSONDecodeError:
                    continue
            else:
                results.append(dict(mode=mode, depth=depth, batch_size=bs, status="error"))

            if results[-1].get("status") == "OOM":
                break  # stop this mode at first OOM

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  SWEEP RESULTS — d{depth} on {torch.cuda.get_device_name(0)}")
    print(f"{'='*70}")
    print(f"  {'Mode':<12} {'Batch':<7} {'Peak VRAM':<12} {'Throughput':<15} {'Status'}")
    print(f"  {'-'*12} {'-'*7} {'-'*12} {'-'*15} {'-'*7}")
    for r in results:
        vram = f"{r.get('peak_vram_gb', '?')} GB" if r['status'] != 'error' else '?'
        tps = f"{r.get('avg_tok_per_s', 0):,} tok/s" if r['status'] == 'ok' else r['status']
        print(f"  {r['mode']:<12} {r['batch_size']:<7} {vram:<12} {tps:<15} {r['status']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matepoint benchmark on nanochat")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--mode", choices=["baseline", "matepoint", "both"], default="both")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sweep", action="store_true", help="sweep batch sizes for both modes")
    parser.add_argument("--json", action="store_true", help="output JSON on last line (for sweep)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.depth, args.seq_len, args.steps, args.gpu)
        sys.exit(0)

    if args.mode == "both":
        # Run each in a subprocess for clean GPU state
        import subprocess
        all_results = []
        for mode in ["baseline", "matepoint"]:
            cmd = [sys.executable, __file__,
                   f"--depth={args.depth}", f"--batch-size={args.batch_size}",
                   f"--seq-len={args.seq_len}", f"--steps={args.steps}",
                   f"--gpu={args.gpu}", f"--mode={mode}", "--json"]
            subprocess.run(cmd)
            print()
    else:
        use_mp = args.mode == "matepoint"
        result = run_benchmark(args.depth, args.batch_size, args.seq_len, args.steps,
                               use_mp, args.gpu)
        if args.json:
            print(json.dumps(result))
