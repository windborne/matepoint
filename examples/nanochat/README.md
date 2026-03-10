# Matepoint × nanochat

Demonstrates matepoint's activation offloading on [nanochat](https://github.com/karpathy/nanochat), Karpathy's minimal LLM training harness.

## Why nanochat?

nanochat is widely used by the LLM community to train GPT-2 class models. It keeps all activations in GPU VRAM with no gradient checkpointing, making activation memory the binding constraint — exactly the problem matepoint solves.

## Results (RTX 4090, 24 GB)

### d12 — 286M params

| Batch Size | Baseline | Matepoint | VRAM Saved |
|---|---|---|---|
| 8 | 16.04 GB @ 81k tok/s | 10.63 GB @ 64k tok/s | 5.4 GB (34%) |
| 16 | OOM | 18.99 GB @ 64k tok/s | **enables 2x batch** |

### d20 — 897M params

| Batch Size | Baseline | Matepoint | VRAM Saved |
|---|---|---|---|
| 4 | 20.03 GB @ 28k tok/s | 12.04 GB @ 21k tok/s | 8.0 GB (40%) |
| 8 | OOM | 16.48 GB @ 22k tok/s | **enables 2x batch** |

**Throughput overhead:** ~22% from checkpoint recomputation + CPU↔GPU transfer. Matepoint pipelines these transfers with computation to minimize the cost.

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
