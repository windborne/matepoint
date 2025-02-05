import torch
import matepoint
import time
def get_gpu_memory():
    """Helper function to get current GPU memory usage"""
    if torch.cuda.is_available():
        return f"{torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB"
    return "Not available"

def create_model():
    """Create a large neural network model"""
    return torch.nn.Sequential(
        torch.nn.Linear(1000, 10000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 10000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(10000, 100)
    )

def create_dummy_data():
    """Create large dummy input data"""
    return torch.randn(10000, 1000)

def test_model(checkpoint_type="none"):
    model = create_model().cuda()
    input = create_dummy_data().cuda()
    input.requires_grad = True  # Ensure input requires gradients
    target = torch.randn(10000, 100).cuda()

    # Initialize
    print(f"Initial GPU memory: {get_gpu_memory()}")
    
    # Define the function to checkpoint
    def create_func(input):
        return model(input)

    if checkpoint_type == "none":
        output = model(input)
    else:
        output = matepoint.checkpoint(create_func, input, use_reentrant=False)
    
    print(f"GPU memory after forward: {get_gpu_memory()}")
    
    # Compute loss and run backward
    loss = torch.nn.MSELoss()(output, target)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    print(f"GPU memory before backward: {get_gpu_memory()}")
    loss.backward()
    print(f"GPU memory after backward: {get_gpu_memory()}")
    del loss, output, input, target, model, create_func
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available. Tests require a GPU to run."
    print("Testing with no checkpointing")
    test_model()
    print("\nTesting with checkpointing")
    test_model("matepoint")
