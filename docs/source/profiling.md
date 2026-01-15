# PyTorch Profiling

We use PyTorch's built-in profiler with `torch-tb-profiler` for TensorBoard visualization.

## Basic Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = YourModel()
inputs = torch.randn(batch_size, ...)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

## TensorBoard Integration

```python
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU],
    on_trace_ready=tensorboard_trace_handler("./log/profiling")
) as prof:
    for i in range(10):
        model(inputs)
        prof.step()
```

Launch TensorBoard:

```bash
tensorboard --logdir=./log
```

Open `http://localhost:6006/#pytorch_profiler` in your browser.
