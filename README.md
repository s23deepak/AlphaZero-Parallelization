
# AlphaZero Parallelization

This repository explores various approaches to parallelizing AlphaZero training for custom turn-based strategy games. The goal is to maximize hardware utilization and accelerate training by distributing self-play and training tasks across multiple CPU and GPU resources.

## Current Implementations

### 1. Ray-based Parallelization (in `alphazero_ray/`)
- **Self-play** is parallelized across 4 CPU cores using [Ray](https://docs.ray.io/en/latest/).
- **MCTS** (Monte Carlo Tree Search) runs on the CPU and is the main bottleneck.
- **Neural network inference and training** are performed on the GPU.
- **Resource allocation**: Each self-play task is assigned one CPU core and a fraction of the GPU (e.g., 0.1 GPU per task).
- **Utilization**: This approach maximizes CPU usage and allows multiple tasks to share the GPU, increasing overall hardware utilization.

### 2. JAX-based Implementation (in `alphazero_jax/`)
- **High-performance computation** using [JAX](https://jax.readthedocs.io/en/latest/) and [MCTX](https://github.com/google-deepmind/mctx) for neural network operations, training, and Monte Carlo Tree Search.
- **JIT compilation** accelerates inference and training on GPU/TPU by compiling Python functions to optimized machine code.
- **Vectorization and batching** with `vmap` and `pmap` to parallelize MCTS simulations and neural network calls across multiple states/devices.
- **Hardware utilization**: JAX maximizes GPU/TPU usage through efficient batch processing and eliminates Python overhead in critical paths.
- **Scalability**: Easily scales to multiple GPUs or TPUs using JAX's distributed primitives.

## Planned Extensions

- **Scalability**: Plans to scale training on cloud platforms (e.g., Amazon SageMaker, Google Vertex AI).
