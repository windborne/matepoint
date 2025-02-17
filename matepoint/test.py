import torch
import matepoint
import time
from torch.utils.checkpoint import checkpoint

def get_gpu_memory():
    """Helper function to get current GPU memory usage"""
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    return "Not available"

class TestModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(1000, 10000)
        self.linear2 = torch.nn.Linear(10000, 10000)
        self.linear3 = torch.nn.Linear(10000, 100)
        self.inplace = inplace
        self.weight = torch.nn.Parameter(torch.randn(1000, 1000))
    
    # def forward(self, x):
    #     x = self.linear1(x)
        
    #     x = torch.relu(x)
    #     x = self.linear2(x)
        
    #     if self.inplace:
    #         # This inplace operation modifies a tensor that was used in computing gradients
    #         x[0, 0] = x[0, 0] ** 2
            
    #     x = torch.relu(x)
    #     x = self.linear3(x)
    #     return x

    def forward(self, x):
        intermediate = torch.matmul(x, self.weight)
        
        # Non-linear operation that will make gradients depend on input values
        y = torch.sin(intermediate)
        
        if self.inplace:
            # Inplace modification of a tensor that's part of gradient computation
            intermediate.add_(1.0)  # This should break gradient computation
            intermediate[0] = intermediate[1]
            
        # Use the modified tensor again
        output = torch.matmul(y, intermediate.t())
        output = output[:, :100]
        return output
    
def create_dummy_data():
    """Create large dummy input data"""
    return torch.randn(10000, 1000)

def test_model(checkpoint_type="none", compile=False, inplace=False):
    print(f"\nTesting with checkpoint type: {checkpoint_type}, compile: {compile}, inplace: {inplace}")
    
    model = TestModel(inplace=inplace).cuda()
    if compile:
        model = torch.compile(model)
    input = create_dummy_data().cuda()
    input.requires_grad = True  # Ensure input requires gradients
    target = torch.randn(10000, 100).cuda()

    # Initialize
    print(f"Initial GPU memory: {get_gpu_memory()}")
    
    # Define the function to checkpoint
    def create_func(input):
        return model(input)

    t1 = time.time()
    if checkpoint_type == "none":
        output = model(input)
    elif checkpoint_type == "matepoint":
        output = matepoint.checkpoint(create_func, input, use_reentrant=False)
    elif checkpoint_type == "checkpoint":
        output = checkpoint(create_func, input, use_reentrant=False)
    else:
        raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")
        
    # Compute loss and run backward
    loss = torch.nn.MSELoss()(output, target)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    t2 = time.time()
    print(f"Time taken for forward: {t2 - t1:.2f} seconds")
    
    print(f"GPU memory after forward: {get_gpu_memory()}")
    loss.backward()
    t3 = time.time()
    print(f"Time taken for backward: {t3 - t2:.2f} seconds")
    print(f"GPU memory after backward: {get_gpu_memory()}")
    del loss, output, input, target, model, create_func
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
    # test_model(checkpoint_type="matepoint")
    # test_model(checkpoint_type="matepoint", compile=True)
    # test_model(checkpoint_type="checkpoint", inplace=True)
    test_nested_matepoint()