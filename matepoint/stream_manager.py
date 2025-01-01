import torch

class StreamManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.stream = torch.cuda.Stream()
        return cls._instance
