import sys
import torch
import matepoint
import time
from torch.utils.checkpoint import checkpoint

def get_gpu_memory():
    """Helper function to get current GPU memory usage"""
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    return "Not available"

def get_TestModel():
    return torch.nn.Sequential(
        torch.nn.Linear(1000, 10000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 10000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 100)
    )

class InPlaceModel(torch.nn.Module):
    def forward(self, x):
        #x = x.clone()
        x[0] += x[1]
        output = torch.sin(x)        
        return output

def test_model(checkpoint_type="none", compile=False, inplace=False):
    print(f"\nTesting with checkpoint type: {checkpoint_type}, compile: {compile}, inplace: {inplace}")
    
    if inplace:
        model = InPlaceModel().cuda()
    else:
        model = get_TestModel().cuda()
    if compile:
        model = torch.compile(model)
    input = torch.randn(10000, 1000, requires_grad=True).cuda() # , device="cuda" gives the leaf node error
    print(f"Initial GPU memory: {get_gpu_memory()}")
    
    def create_func(input):
        return model(input)

    t1 = time.time()
    if checkpoint_type == "none":
        output = model(input)
    elif checkpoint_type == "matepoint":
        output = matepoint.checkpoint(create_func, input, use_reentrant=False)
    elif checkpoint_type == "torch":
        output = checkpoint(create_func, input, use_reentrant=False)
    else:
        raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")
        
    loss = output.sum()
    t2 = time.time()
    print(f"Time taken for forward: {t2 - t1:.2f} seconds")
    print(f"GPU memory after forward: {get_gpu_memory()}")
    
    loss.backward()
    t3 = time.time()
    print(f"Time taken for backward: {t3 - t2:.2f} seconds")
    print(f"GPU memory after backward: {get_gpu_memory()}")
    del loss, output, input, model, create_func
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

class NestedModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
    
    def forward(self, x):
        def nested_func(x):
            return matepoint.checkpoint(self.linear, x, use_reentrant=False)
        
        # This should raise an error due to nested checkpoint
        return matepoint.checkpoint(nested_func, x, use_reentrant=False)

def test_nested_matepoint():
    print("\nTesting nested matepoint calls")
    
    model = NestedModule().cuda()
    input = torch.randn(32, 100, requires_grad=True).cuda()
    
    try:
        output = model(input)
        loss = output.sum()
        loss.backward()
        print("✓ Nested matepoint calls succeeded")
    except Exception as e:
        print(f"✗ Nested matepoint calls failed with error: {str(e)}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Tests require a GPU to run."
    #test_model(checkpoint_type="matepoint")
    # test_model(checkpoint_type="matepoint", compile=True)
    # take command line arg for checkpoint type
    test_model(checkpoint_type='matepoint', inplace=True)
    #test_nested_matepoint()