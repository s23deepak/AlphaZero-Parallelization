"""
Training script for AlphaZero on AaduPulliPGXEnv

This script sets up the training configuration and provides a framework
for training an AlphaZero agent on the custom environment.
"""

import jax
import jax.numpy as jnp
from aadupulli_env import AaduPulliPGXEnv

def create_alphazero_config():
    """Create configuration for AlphaZero training."""
    config = {
        'env': AaduPulliPGXEnv(),
        'num_simulations': 50,
        'num_training_steps': 1000,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'model_type': 'mlp',  # or 'resnet' if supported
        'c_puct': 1.0,  # UCB constant
        'dirichlet_alpha': 0.3,  # Dirichlet noise parameter
        'exploration_fraction': 0.25,  # Fraction of moves with exploration noise
        'temperature': 1.0,  # Temperature for action selection
        'max_game_length': 200,
        'checkpoint_interval': 100,
        'evaluation_games': 10,
    }
    return config

def test_environment():
    """Test the environment to ensure it works correctly."""
    print("Testing AaduPulliPGXEnv...")
    
    env = AaduPulliPGXEnv()
    key = jax.random.PRNGKey(42)
    
    # Initialize environment
    state = env.init(key)
    print(f"Initial state - Current player: {state.current_player}")
    print(f"Initial board: {state.board}")
    print(f"Goats to place: {state.goats_to_place}")
    print(f"Legal actions count: {jnp.sum(state.legal_action_mask)}")
    
    # Test a few steps
    legal_actions = jnp.where(state.legal_action_mask)[0]
    if len(legal_actions) > 0:
        # Take a goat placement action
        action = legal_actions[0]
        print(f"Taking action: {action}")
        
        state = env.step(state, action, key)
        print(f"After action - Current player: {state.current_player}")
        print(f"Board after action: {state.board}")
        print(f"Game terminated: {state.terminated}")
        print(f"Rewards: {state.rewards}")
    
    print("Environment test completed successfully!")
    return True

def simulate_tiger_win():
    """Simulate a scenario where tiger wins to test reward system."""
    print("\n--- Testing Tiger Win Scenario ---")
    
    env = AaduPulliPGXEnv()
    key = jax.random.PRNGKey(123)
    
    # Create a state where tiger is about to win (9 goats captured)
    initial_state = env.init(key)
    
    # Manually set goats_captured to be one less than threshold
    state = initial_state.replace(goats_captured=jnp.int32(9))
    
    # Set up board for a tiger jump that captures the 10th goat
    # Tiger at pos 1, goat at pos 3, tiger jumps to pos 9
    test_board = state.board.at[2].set(1)  # Place goat at position 3 (index 2)
    state = state.replace(board=test_board, current_player=jnp.int32(1))  # Tiger's turn
    
    print(f"Before tiger win - Goats captured: {state.goats_captured}")
    
    # Find the jump action (Tiger from pos 1 to pos 9, jumping over pos 3)
    from aadupulli_env import MOVE_INFO, PLACEMENT_ACTIONS
    jump_move_idx = None
    for i, (from_pos, to_pos, is_jump, mid_pos) in enumerate(MOVE_INFO):
        if from_pos == 1 and to_pos == 9 and is_jump == 1 and mid_pos == 3:
            jump_move_idx = i
            break
    
    if jump_move_idx is not None:
        action = jnp.int32(PLACEMENT_ACTIONS + jump_move_idx)
        print(f"Tiger jump action: {action}")
        
        state = env.step(state, action, key)
        
        print(f"After tiger win - Goats captured: {state.goats_captured}")
        print(f"Game terminated: {state.terminated}")
        print(f"Rewards: {state.rewards}")
        
        # Verify the fix: Tiger win should give Goat -1, Tiger +1
        expected_rewards = jnp.array([-1.0, 1.0])
        if jnp.allclose(state.rewards, expected_rewards):
            print("✅ SUCCESS: Reward system is working correctly!")
        else:
            print(f"❌ FAILURE: Expected rewards {expected_rewards}, got {state.rewards}")
            return False
    else:
        print("Could not find the tiger jump action")
        return False
    
    return True

def main():
    """Main training function."""
    print("AlphaZero Training for AaduPulliPGXEnv")
    print("=" * 50)
    
    # Test the environment first
    if not test_environment():
        print("Environment test failed!")
        return
    
    # Test the reward system fix
    if not simulate_tiger_win():
        print("Reward system test failed!")
        return
    
    # Create training configuration
    config = create_alphazero_config()
    print(f"\nTraining configuration:")
    for key, value in config.items():
        if key != 'env':  # Don't print the environment object
            print(f"  {key}: {value}")
    
    print("\nEnvironment is ready for AlphaZero training!")
    print("Note: This script provides the framework. Full AlphaZero training would require")
    print("additional components like MCTS, neural network models, and training loops.")
    print("\nTo integrate with existing AlphaZero implementations:")
    print("1. Import AaduPulliPGXEnv from aadupulli_env")
    print("2. Use the configuration provided by create_alphazero_config()")
    print("3. Ensure the training framework can handle PGX environments")

if __name__ == "__main__":
    main()