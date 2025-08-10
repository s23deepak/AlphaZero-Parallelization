
# AlphaZero with Ray-based Parallelization

This directory contains an implementation of AlphaZero for a custom turn-based strategy game, using distributed self-play and training with [Ray](https://docs.ray.io/en/latest/). Monte Carlo Tree Search (MCTS) is combined with a convolutional neural network (CNN) for policy and value prediction. Ray is used to parallelize self-play games across multiple CPU cores while sharing GPU resources for neural network inference and training.


## How It Works

- **MCTS** is performed on the CPU and is the main bottleneck in AlphaZero training.
- **Neural network inference and training** are performed on the GPU.
- **Ray** is used to launch multiple self-play tasks in parallel, each running its own MCTS and periodically querying the neural network.


### Ray Remote Decorator

The following line is key to distributing self-play tasks:


```python
@ray.remote(num_cpus=1, num_gpus=0.1)
def self_play_task(model_weights):
    ...
```


- `num_cpus=1`: Each self-play task is allocated one CPU core.
- `num_gpus=0.1`: Each task is allocated 10% of a GPU, allowing up to 10 tasks to share a single GPU.


#### Effect on Utilization

- **CPU Utilization**: By setting `num_cpus=1` and launching as many tasks as there are CPU cores, you maximize CPU usage. For example, on a 4-core machine, running 4 tasks in parallel can lead to ~300% CPU utilization (since each task is CPU-bound due to MCTS).
- **GPU Utilization**: Since each task only occasionally calls the neural network for inference, the GPU is underutilized (e.g., ~3% usage). Increasing the number of parallel tasks (by reducing `num_gpus` per task or increasing the number of CPU cores) can increase GPU utilization, as more tasks will be making inference calls concurrently.

**Note:** If you set `num_gpus=0.05`, you could run up to 20 tasks per GPU, further increasing GPU usage, provided you have enough CPU cores to support more tasks.


## Observations

- **Bottleneck**: MCTS is sequential and CPU-bound, so the CPU is the limiting factor.
- **GPU Underutilization**: The neural network is only queried a few times per game, so the GPU is mostly idle unless many games are run in parallel.
- **Distributed Training**: Ray helps bridge the gap by allowing multiple CPU-bound tasks to share the GPU, increasing overall hardware utilization.