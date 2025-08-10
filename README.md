
# AlphaZero Parallelization

This repository explores various approaches to parallelizing AlphaZero training for custom turn-based strategy games. The goal is to maximize hardware utilization and accelerate training by distributing self-play and training tasks across multiple CPU and GPU resources.

## Current Implementations

### 1. Ray-based Parallelization (in `alphazero_ray/`)
- **Self-play** is parallelized across 4 CPU cores using [Ray](https://docs.ray.io/en/latest/).
- **MCTS** (Monte Carlo Tree Search) runs on the CPU and is the main bottleneck.
- **Neural network inference and training** are performed on the GPU.
- **Resource allocation**: Each self-play task is assigned one CPU core and a fraction of the GPU (e.g., 0.1 GPU per task).
- **Utilization**: This approach maximizes CPU usage and allows multiple tasks to share the GPU, increasing overall hardware utilization.

## Planned Extensions

- **Numba/JAX Acceleration**: Future implementations will explore using [Numba](https://numba.pydata.org/) or [JAX](https://jax.readthedocs.io/) to accelerate MCTS or other components on the GPU.
- **Scalability**: Plans to scale training on cloud platforms (e.g., Amazon SageMaker, Google Vertex AI).