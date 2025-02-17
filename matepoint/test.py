import torch
import matepoint
import time
def get_gpu_memory():
    """Helper function to get current GPU memory usage"""
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    return "Not available"

def create_model(inplace=False):
    """Create a large neural network model"""
    return torch.nn.Sequential(
        torch.nn.Linear(1000, 10000),
        torch.nn.ReLU() if not inplace else torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 10000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 100)
    )

def create_dummy_data():
    """Create large dummy input data"""
    return torch.randn(10000, 1000)

def test_model(checkpoint_type="none", compile=False):
    print(f"\nTesting with checkpoint type: {checkpoint_type} and compile: {compile}")
    
    model = create_model().cuda()
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
        output = torch.utils.checkpoint.checkpoint(create_func, input, use_reentrant=False)
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

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Tests require a GPU to run."
    test_model(checkpoint_type="matepoint")
    test_model(checkpoint_type="matepoint", compile=True)
    # test_model(checkpoint_type="matepoint")