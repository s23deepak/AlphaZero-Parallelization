#!/usr/bin/env python3
"""Test script to verify PGX environment vectorization."""

import jax
import jax.numpy as jnp
from jax import random, vmap
import sys

# Import the environment from the main file
# The file is named with parentheses and spaces, so we need to import it properly
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("apa_module", "apa_jax_e2e_(3).py")
apa_module = importlib.util.module_from_spec(spec)
sys.modules["apa_module"] = apa_module
spec.loader.exec_module(apa_module)

AaduPuliAattamJAX = apa_module.AaduPuliAattamJAX

def test_single_env():
    """Test single environment initialization and step."""
    print("=" * 60)
    print("Test 1: Single Environment")
    print("=" * 60)
    
    env = AaduPuliAattamJAX()
    key = random.PRNGKey(42)
    
    # Initialize environment
    state = env.init(key)
    
    print(f"✓ Environment initialized")
    print(f"  - Observation shape: {state.observation.shape}")
    print(f"  - Number of actions: {env.num_actions}")
    print(f"  - Current player: {state.current_player}")
    print(f"  - Legal actions available: {jnp.sum(state.legal_action_mask)}")
    print(f"  - Terminated: {state.terminated}")
    
    # Take a step with a legal action
    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        action = legal_actions[0]
        next_state = env.step(state, action)
        print(f"\n✓ Step executed with action {action}")
        print(f"  - Next player: {next_state.current_player}")
        print(f"  - Rewards: {next_state.rewards}")
        print(f"  - Terminated: {next_state.terminated}")
        print(f"  - Legal actions available: {jnp.sum(next_state.legal_action_mask)}")
    
    print("\n✅ Single environment test passed!\n")


def test_vectorized_env():
    """Test vectorized environment initialization and steps."""
    print("=" * 60)
    print("Test 2: Vectorized Environment (32 parallel games)")
    print("=" * 60)
    
    env = AaduPuliAattamJAX()
    batch_size = 32
    
    # Create keys for batch initialization
    key = random.PRNGKey(123)
    keys = random.split(key, batch_size)
    
    # Vectorized init
    print(f"Initializing {batch_size} environments in parallel...")
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    print(f"✓ {batch_size} environments initialized")
    print(f"  - Observation shape: {states.observation.shape}")
    print(f"  - Legal action mask shape: {states.legal_action_mask.shape}")
    print(f"  - Current players shape: {states.current_player.shape}")
    print(f"  - Average legal actions per env: {jnp.mean(jnp.sum(states.legal_action_mask, axis=1)):.1f}")
    
    # Vectorized step
    print(f"\nExecuting vectorized steps...")
    
    # Select first legal action for each environment
    def get_first_legal_action(mask):
        legal_actions = jnp.where(mask, size=1, fill_value=0)[0]
        return legal_actions[0]
    
    actions = vmap(get_first_legal_action)(states.legal_action_mask)
    
    # Execute vectorized step
    step_fn = jax.vmap(env.step)
    next_states = step_fn(states, actions)
    
    print(f"✓ {batch_size} steps executed in parallel")
    print(f"  - Next players shape: {next_states.current_player.shape}")
    print(f"  - Rewards shape: {next_states.rewards.shape}")
    print(f"  - Terminated games: {jnp.sum(next_states.terminated)}")
    print(f"  - Average legal actions per env: {jnp.mean(jnp.sum(next_states.legal_action_mask, axis=1)):.1f}")
    
    print("\n✅ Vectorized environment test passed!\n")


def test_full_game():
    """Test a full game rollout with vectorized environments."""
    print("=" * 60)
    print("Test 3: Full Game Rollout (8 parallel games, 20 steps)")
    print("=" * 60)
    
    env = AaduPuliAattamJAX()
    batch_size = 8
    num_steps = 20
    
    key = random.PRNGKey(456)
    keys = random.split(key, batch_size)
    
    # Initialize
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    print(f"Starting {batch_size} parallel games...")
    
    # Simulate random play for num_steps
    step_fn = jax.vmap(env.step)
    
    def select_random_legal_action(key, mask):
        legal_indices = jnp.where(mask, size=env.num_actions, fill_value=-1)[0]
        valid_count = jnp.sum(mask)
        # Select random index among legal actions
        random_idx = random.randint(key, (), 0, jnp.maximum(valid_count, 1))
        action = jnp.where(valid_count > 0, legal_indices[random_idx], 0)
        return action
    
    for step in range(num_steps):
        # Select random legal actions
        step_keys = random.split(key, batch_size)
        key = random.split(key)[0]
        actions = vmap(select_random_legal_action)(step_keys, states.legal_action_mask)
        
        # Step all environments
        states = step_fn(states, actions)
        
        terminated_count = jnp.sum(states.terminated)
        if step % 5 == 0 or step == num_steps - 1:
            print(f"  Step {step+1:2d}: {terminated_count}/{batch_size} games finished")
        
        if jnp.all(states.terminated):
            print(f"\n✓ All games finished at step {step+1}")
            break
    
    # Final statistics
    print(f"\nFinal statistics:")
    print(f"  - Finished games: {jnp.sum(states.terminated)}/{batch_size}")
    print(f"  - Goat wins: {jnp.sum(states.rewards[:, 0] > 0)}")
    print(f"  - Tiger wins: {jnp.sum(states.rewards[:, 1] > 0)}")
    print(f"  - Draws: {jnp.sum((states.rewards[:, 0] == 0) & states.terminated)}")
    
    print("\n✅ Full game rollout test passed!\n")


def benchmark_vectorization():
    """Benchmark vectorized vs sequential execution."""
    print("=" * 60)
    print("Test 4: Vectorization Benchmark")
    print("=" * 60)
    
    import time
    
    env = AaduPuliAattamJAX()
    batch_size = 128
    num_steps = 10
    
    key = random.PRNGKey(789)
    
    # Vectorized execution
    print(f"Benchmarking vectorized execution ({batch_size} envs, {num_steps} steps)...")
    keys = random.split(key, batch_size)
    
    start = time.time()
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    step_fn = jax.vmap(env.step)
    for _ in range(num_steps):
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        states = step_fn(states, actions)
    
    # Force computation
    jax.block_until_ready(states)
    vectorized_time = time.time() - start
    
    print(f"✓ Vectorized execution time: {vectorized_time:.4f}s")
    print(f"  ({batch_size * num_steps / vectorized_time:.1f} env-steps/sec)")
    
    # Sequential execution (for comparison)
    print(f"\nBenchmarking sequential execution ({batch_size} envs, {num_steps} steps)...")
    
    start = time.time()
    for i in range(batch_size):
        state = env.init(keys[i])
        for _ in range(num_steps):
            action = jnp.int32(0)
            state = env.step(state, action)
        jax.block_until_ready(state)
    
    sequential_time = time.time() - start
    
    print(f"✓ Sequential execution time: {sequential_time:.4f}s")
    print(f"  ({batch_size * num_steps / sequential_time:.1f} env-steps/sec)")
    
    speedup = sequential_time / vectorized_time
    print(f"\n🚀 Speedup from vectorization: {speedup:.2f}x")
    
    print("\n✅ Benchmark completed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PGX Environment Vectorization Tests")
    print("=" * 60 + "\n")
    
    try:
        test_single_env()
        test_vectorized_env()
        test_full_game()
        benchmark_vectorization()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour PGX environment is ready for high-performance training!")
        print("The vectorization enables parallel simulation of many games,")
        print("which is crucial for efficient AlphaZero training.\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
