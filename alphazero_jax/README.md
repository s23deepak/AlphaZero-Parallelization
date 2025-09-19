# AlphaZero with JAX-based Implementation

This directory contains an implementation of AlphaZero for a custom turn-based strategy game, using [JAX](https://jax.readthedocs.io/en/latest/) for high-performance computation and parallelization. Monte Carlo Tree Search (MCTS) is combined with a convolutional neural network (CNN) for policy and value prediction. JAX is used to accelerate neural network operations and enable efficient parallelization across devices like GPUs and TPUs.

## How It Works

- **JAX** provides just-in-time (JIT) compilation and automatic differentiation to speed up computations and enable efficient batching and parallelization.
- **MCTX** By [DeepMind](https://github.com/google-deepmind/mctx), is Monte Carlo Tree Search implementation in JAX
- **Neural network inference and training** are performed on the GPU/TPU using JAX's optimized primitives.

### JAX JIT and Vectorization

- `@jax.jit`: Compiles the function for efficient execution on CPU/GPU/TPU, eliminating Python overhead.
- `jax.vmap`: Automatically vectorizes the function to handle batches of inputs efficiently, enabling parallel processing of multiple game states.

#### Effect on Utilization

- **CPU Utilization**: MCTS remains CPU-bound, but JAX's JIT compilation reduces overhead in neural network calls.
- **GPU/TPU Utilization**: JAX enables high utilization by compiling operations to optimized kernels and supporting batch processing. Multiple MCTS simulations can be batched together for inference, improving hardware efficiency.
- **Parallelization**: JAX's `pmap` can distribute computations across multiple devices if available, further increasing throughput.

**Note:** JAX's functional programming paradigm requires careful state management and [pure functions](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions), but it provides significant performance gains for numerical computations in AlphaZero.

## Observations

- **GPU/TPU Utilization**: JAX maximizes hardware utilization through compilation and vectorization, making neural network operations much faster than in pure NumPy implementations.
- **Scalability**: JAX's design allows for easy scaling to multiple GPUs or TPUs, providing better performance than CPU-only implementations.