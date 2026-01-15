import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {name}")
else:
    print("No GPUs available.")
