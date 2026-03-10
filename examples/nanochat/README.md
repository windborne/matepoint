# Matepoint × nanochat

Demonstrates matepoint's activation offloading on [nanochat](https://github.com/karpathy/nanochat), Karpathy's minimal LLM training harness.

## Why nanochat?

nanochat is widely used by the LLM community to train GPT-2 class models. It keeps all activations in GPU VRAM with no gradient checkpointing, making activation memory the binding constraint — exactly the problem matepoint solves.

## Results (RTX 4090, 24 GB)

### d12 — 286M params

| Batch | Mode | Peak VRAM | Step Time | Throughput | CPU RAM delta |
|---|---|---|---|---|---|
| 8 | Baseline | 16.04 GB | 202 ms | 81k tok/s | +0.4 GB |
| 8 | Matepoint | **10.63 GB** | 258 ms | 64k tok/s | +0.9 GB |
| 16 | Baseline | OOM | — | — | — |
| 16 | Matepoint | **18.99 GB** | 513 ms | 64k tok/s | +1.5 GB |

### d20 — 897M params

| Batch | Mode | Peak VRAM | Step Time | Throughput | CPU RAM delta |
|---|---|---|---|---|---|
| 4 | Baseline | 20.03 GB | 295 ms | 28k tok/s | +0.4 GB |
| 4 | Matepoint | **12.04 GB** | 383 ms | 21k tok/s | +1.3 GB |
| 8 | Baseline | OOM | — | — | — |
| 8 | Matepoint | **16.48 GB** | 746 ms | 22k tok/s | +2.3 GB |

**At same batch size:** 34–40% VRAM reduction with ~28% step time overhead and only ~1–2 GB extra CPU RAM. Matepoint pipelines CPU↔GPU transfers with computation to minimize the throughput cost.

> **Note on throughput overhead:** The ~28% overhead is inflated here because nanochat's small models have very short step times (~200–300 ms), leaving little computation to hide PCIe transfers behind. With larger models where step times are in seconds, the async pipelining fully overlaps transfers with compute and overhead drops to near-zero. ~15% of the overhead is also from gradient recomputation (inherent to checkpointing, not matepoint-specific).

## How it works

The integration is a single function that wraps each transformer block with `matepoint.checkpoint`:

```python
import matepoint

for i, block in enumerate(model.transformer.h):
    x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
    ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None

    # Instead of: x = block(x, ve, cos_sin, window_size, None)
    x = matepoint.checkpoint(
        block, x, ve, cos_sin, self.window_sizes[i], None,
        use_reentrant=False,
    )
```

During the forward pass, activations for each block are offloaded to CPU RAM. During backward, they're pipelined back to GPU just before they're needed.

## Setup

```bash
# Clone nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat
uv sync --extra gpu
uv pip install matepoint

# Download data and train tokenizer (minimal setup)
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
python -m nanochat.dataset -n 8
python -m scripts.tok_train
```

## Run the benchmark

```bash
# Compare baseline vs matepoint at a specific batch size
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python path/to/benchmark.py --depth=12 --batch-size=8

# Show matepoint enabling a batch size that OOMs without it
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python path/to/benchmark.py --depth=20 --batch-size=8 --mode=matepoint

# Sweep all batch sizes to find the limits
NANOCHAT_BASE_DIR=$HOME/.cache/nanochat python path/to/benchmark.py --depth=12 --sweep
```
