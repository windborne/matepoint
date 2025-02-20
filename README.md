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
    out = checkpoint(self.layer, x)
    return out
```

## Requirements

- PyTorch >= 2.4.0
- CUDA-capable GPU
- Sufficient CPU memory for activation storage

## Installation

```bash
git clone https://github.com/yourusername/matepoint.git
cd matepoint
pip install -e .
```

## References
Refer to the Matepoint section in this [blog post](https://windbornesystems.com/blog/weathermesh-2-technical-blog) for more details on the implementation and performance benefits.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
