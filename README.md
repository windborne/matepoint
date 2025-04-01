# Matepoint


## Overview

Matepoint is a fork of PyTorch's `torch.utils.checkpoint` that allows you to utilize CPU RAM when you're low on GPU VRAM. While standard checkpointing trades computation for memory by recomputing activations during the backward pass, Matepoint takes this further by:

1. Automatically offloading activation tensors to CPU after the forward pass
2. Efficiently moving tensors back to GPU only when needed during the backward pass
3. Supporting pipelined tensor transfers for better performance
4. Providing optional CPU memory pooling for large, similarly-shaped tensors

## Usage

Replace your existing `torch.utils.checkpoint` calls with `matepoint`:

```python
from matepoint import checkpoint

# Instead of:
# from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Use exactly like torch.utils.checkpoint
    out = checkpoint(self.layer, x, use_reentrant=False)
    return out
```

## Requirements

- PyTorch >= 2.4.0
- CUDA-capable GPU
- Sufficient CPU memory for activation storage

## Installation

```bash
pip install --index-url https://test.pypi.org/simple/ matepoint
```

## Build
```bash
python setup.py sdist bdist_wheel
twine upload --repository testpypi dist/*
```

## References
Refer to the Matepoint section in this [blog post](https://windbornesystems.com/blog/weathermesh-2-technical-blog) for more details on the implementation and performance benefits.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Real-World Example

We actually built Matepoint when we were running out of VRAM trying to solve weather(™) with transformers. While WeatherMesh, our model itself isn't huge (~180M parameters), forecasting weather for the entire planet over 6 days means running through 200+ transformer layers.

Without some clever tricks, we'd need hundreds of GiB of VRAM. Even regular checkpointing wasn't enough - storing those 200MiB latent tensors for each transformer block would eat up around 40GiB of VRAM, which is more than even an RTX 4090 can handle.

Matepoint ships those tensors off to CPU RAM when we don't need them, then brings them back just in time during the backward pass. Adding more forecast days costs almost nothing in VRAM terms. This meant we could train our whole weather model on consumer RTX 4090s instead of shelling out for pricier hardware.

Check out these visualizations to see Matepoint in action:

![Matepoint forward pass](images/Matepoint_fw.svg)
![Matepoint backward pass](images/Matepoint_bw.svg)

## Advanced Options

### Pipeline Mode

Matepoint overlaps data movement with computation by default, improving performance by efficiently transferring tensors between CPU and GPU. You can disable this optimization if needed:

```python
import matepoint
matepoint.NOPIPELINE = True  # Disable pipelined tensor transfers
```
