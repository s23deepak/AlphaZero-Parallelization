# AaduPulliPGXEnv - Fixed Implementation

This directory contains the corrected implementation of the AaduPulliPGXEnv (Aadu Puli Aattam game) for AlphaZero training, with the reward system bug fixed.

## Problem Resolved

**Original Issue**: AssertionError during training - "Player 0 (Goat) should have a reward of -1" when the tiger won.

**Root Cause**: The reward calculation logic in the `_step` method was incorrect, giving the wrong rewards when the game terminated.

**Solution**: Fixed the reward assignment to correctly give:
- Tiger win: Goat gets -1, Tiger gets +1 → `[-1.0, 1.0]`
- Goat win: Goat gets +1, Tiger gets -1 → `[1.0, -1.0]`
- Draw: Both get 0 → `[0.0, 0.0]`

## Files

### `aadupulli_env.py`
Complete PGX-compatible environment implementation with:
- ✅ Fixed reward system
- ✅ All required abstract methods (`_init`, `_step`, `_observe`, `_legal_action_mask`)
- ✅ Proper game logic for Aadu Puli Aattam
- ✅ JAX-compatible implementation

### `train.py`
Training script framework that provides:
- Environment testing functionality
- AlphaZero configuration setup
- Tiger win scenario simulation
- Integration instructions

### `test_rewards.py`
Test suite for validating the reward system fix without external dependencies.

### `verify_fix.py`
Specific verification that the AssertionError has been resolved.

### `integration_test.py`
Complete integration test suite verifying all components work together.

## Usage

1. **Install Dependencies:**
   ```bash
   pip install jax jaxlib pgx-game flax
   ```

2. **Import and Use:**
   ```python
   from aadupulli_env import AaduPulliPGXEnv
   
   env = AaduPulliPGXEnv()
   key = jax.random.PRNGKey(42)
   state = env.init(key)
   ```

3. **Training Configuration:**
   ```python
   from train import create_alphazero_config
   config = create_alphazero_config()
   ```

## Game Rules (Aadu Puli Aattam)

- **Players**: 2 (Goat player vs Tiger player)
- **Pieces**: 15 Goats, 3 Tigers
- **Board**: 23 positions with specific adjacency relationships
- **Objective**: 
  - Tigers win by capturing 10 goats
  - Goats win by blocking all tiger movements
- **Phases**:
  1. Placement phase: Goats are placed on empty positions
  2. Movement phase: Both pieces can move to adjacent positions
  3. Tigers can jump over goats to capture them

## Testing

Run the test suite to verify everything works:

```bash
python test_rewards.py      # Test reward system
python verify_fix.py        # Verify AssertionError fix  
python integration_test.py  # Full integration test
```

## Key Fix Details

The critical fix was in the `_step` method around line 165-170:

```python
# FIXED: Calculate reward based on termination condition
reward = jax.lax.cond(
    terminated,
    lambda: jax.lax.cond(
        t_win,
        lambda: jnp.array([-1.0, 1.0]),  # Tiger win: Goat gets -1, Tiger gets +1
        lambda: jax.lax.cond(
            g_win,
            lambda: jnp.array([1.0, -1.0]),  # Goat win: Goat gets +1, Tiger gets -1
            lambda: jnp.zeros(2, dtype=jnp.float32)  # Draw
        )
    ),
    lambda: jnp.zeros(2, dtype=jnp.float32)  # No reward if not terminated
)
```

This ensures that when the tiger wins by capturing 10 goats, the goat player (Player 0) correctly receives a negative reward (-1.0) and the tiger player (Player 1) receives a positive reward (+1.0).

## Status: ✅ COMPLETE

All problem statement requirements have been addressed:
- ✅ AssertionError resolved
- ✅ PGX environment implementation complete
- ✅ Training script provided
- ✅ Reward system fixed and tested
- ✅ Ready for AlphaZero integration