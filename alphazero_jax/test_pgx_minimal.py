#!/usr/bin/env python3
"""Test script to verify PGX environment vectorization - minimal version."""

import jax
import jax.numpy as jnp
from jax import random, vmap, lax
from functools import partial
from typing import Tuple, Dict, NamedTuple
import chex

# Minimal imports from PGX
import pgx

# Copy the environment class definition here to avoid importing the full training script
@chex.dataclass(frozen=True)
class State:
    """PGX-compatible state for the Aadu Puli Aattam game."""
    # PGX required fields
    current_player: jax.Array
    observation: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    legal_action_mask: jax.Array
    _step_count: jax.Array  # Required by PGX (with underscore)
    # Custom fields for our game
    board: jax.Array  # The actual board state
    goats_to_place: jax.Array
    goats_captured: jax.Array
    
    @property
    def env_id(self):
        return "aadu_puli_aattam"

class AaduPuliAattamJAX(pgx.Env):
    def __init__(self):
        super().__init__()
        self.NUM_GOATS = 15
        self.NUM_TIGERS = 3
        self.TIGER_WIN_THRESHOLD = 10
        self.BOARD_POSITIONS = 23
        self.MAX_TURNS = 200

        adj_dict = self._get_adjacency()
        jump_adj_dict = self._get_jump_adjacency()
        self.max_neighbors = max((len(v) for v in adj_dict.values()), default=0)
        self._move_action_map, self._move_action_lookup = self._create_move_maps(adj_dict, jump_adj_dict)
        self.placement_actions_count = self.BOARD_POSITIONS
        self.move_actions_count = len(self._move_action_map)
        self.total_actions = self.placement_actions_count + self.move_actions_count
        self._adj_matrix, self._num_adj = self._create_adj_matrix_from_dict(adj_dict)
        self._jump_adj_matrix, self._num_jump_adj = self._create_adj_matrix_from_dict(jump_adj_dict)

    def _create_adj_matrix_from_dict(self, adj_dict: Dict[int, list]) -> Tuple[chex.Array, chex.Array]:
        import numpy as np
        matrix = np.zeros((self.BOARD_POSITIONS + 1, self.max_neighbors), dtype=np.int32)
        counts = np.zeros(self.BOARD_POSITIONS + 1, dtype=np.int32)
        for pos, neighbors in adj_dict.items():
            num = len(neighbors)
            matrix[pos, :num] = neighbors
            counts[pos] = num
        return jnp.array(matrix), jnp.array(counts)

    def _get_adjacency(self) -> Dict[int, list]:
        return {
            1: [3, 4, 5, 6], 2: [3, 8], 3: [1, 4, 9, 2], 4: [1, 5, 10, 3], 5: [1, 6, 11, 4], 6: [1, 7, 12, 5], 7: [6, 13],
            8: [2, 9, 14], 9: [3, 10, 15, 8], 10: [4, 11, 16, 9], 11: [5, 12, 17, 10], 12: [6, 13, 18, 11], 13: [7, 14, 12],
            14: [8, 15], 15: [9, 16, 20, 14], 16: [10, 17, 21, 15], 17: [11, 18, 22, 16], 18: [12, 19, 23, 17], 19: [13, 18],
            20: [15, 21], 21: [16, 20, 22], 22: [17, 21, 23], 23: [18, 22]
        }

    def _get_jump_adjacency(self) -> Dict[int, list]:
        return {
            1: [9, 10, 11, 12], 2: [4, 14], 3: [5, 15], 4: [2, 6, 16], 5: [3, 7, 17], 6: [4, 18], 7: [5, 19],
            8: [10], 9: [1, 11, 20], 10: [1, 8, 12, 21], 11: [1, 9, 13, 22], 12: [1, 10, 23], 13: [11],
            14: [2, 16], 15: [3, 17], 16: [4, 14, 18], 17: [5, 15, 19], 18: [6, 16], 19: [7, 17],
            20: [9, 22], 21: [10, 23], 22: [11, 20], 23: [12, 21]
        }

    def _create_move_maps(self, adj_dict: Dict, jump_adj_dict: Dict) -> Tuple[chex.Array, dict]:
        action_map_list = []
        action_lookup = {}
        index = 0
        for start_pos in range(1, self.BOARD_POSITIONS + 1):
            for end_pos in adj_dict.get(start_pos, []):
                move = (start_pos, end_pos)
                action_map_list.append(move)
                action_lookup[move] = index
                index += 1
            for end_pos in jump_adj_dict.get(start_pos, []):
                move = (start_pos, end_pos)
                if move not in action_lookup:
                    action_map_list.append(move)
                    action_lookup[move] = index
                    index += 1
        return jnp.array(action_map_list, dtype=jnp.int32), action_lookup

    def _init(self, key: jax.Array) -> State:
        board = jnp.zeros(self.BOARD_POSITIONS, dtype=jnp.int32)
        board = board.at[jnp.array([0, 3, 4])].set(2)
        
        init_state = State(
            current_player=jnp.int32(0),
            observation=self._make_observation(board, 0, self.NUM_GOATS, 0, 0),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=jnp.zeros(self.total_actions, dtype=jnp.bool_),
            _step_count=jnp.int32(0),
            board=board,
            goats_to_place=jnp.int32(self.NUM_GOATS),
            goats_captured=jnp.int32(0)
        )
        
        legal_mask = self._compute_legal_actions(init_state)
        return init_state.replace(legal_action_mask=legal_mask)

    def _compute_legal_actions(self, state: State) -> chex.Array:
        mask = jnp.zeros(self.total_actions, dtype=jnp.bool_)

        def goat_placement_mask():
            return (state.board == 0).astype(jnp.bool_)

        def goat_move_mask():
            move_mask = jnp.zeros(self.move_actions_count, dtype=jnp.bool_)
            goat_indices = jnp.where(state.board == 1, size=self.NUM_GOATS, fill_value=-1)[0]

            def check_move(i, current_mask):
                from_idx = goat_indices[i]
                from_pos = from_idx + 1

                def update_mask(j, inner_mask):
                    end_pos = self._adj_matrix[from_pos, j]
                    to_idx = end_pos - 1
                    is_empty = (state.board[to_idx] == 0)
                    move_tuple = jnp.array([from_pos, end_pos])
                    action_idx = jnp.where(jnp.all(self._move_action_map == move_tuple, axis=1), size=1, fill_value=-1)[0][0]
                    return lax.cond(
                        (action_idx != -1) & is_empty,
                        lambda: inner_mask.at[action_idx].set(True),
                        lambda: inner_mask
                    )
                num_neighbors = self._num_adj[from_pos]
                return lax.fori_loop(0, num_neighbors, update_mask, current_mask)

            return lax.fori_loop(0, self.NUM_GOATS, check_move, move_mask)

        def tiger_move_mask():
            move_mask = jnp.zeros(self.move_actions_count, dtype=jnp.bool_)
            tiger_indices = jnp.where(state.board == 2, size=self.NUM_TIGERS, fill_value=-1)[0]

            def check_tiger_moves(i, current_mask):
                from_idx = tiger_indices[i]
                from_pos = from_idx + 1

                def check_adj_move(j, inner_mask):
                    end_pos = self._adj_matrix[from_pos, j]
                    to_idx = end_pos - 1
                    is_empty = (state.board[to_idx] == 0)
                    move_tuple = jnp.array([from_pos, end_pos])
                    action_idx = jnp.where(jnp.all(self._move_action_map == move_tuple, axis=1), size=1, fill_value=-1)[0][0]
                    return lax.cond(
                        (action_idx != -1) & is_empty,
                        lambda: inner_mask.at[action_idx].set(True),
                        lambda: inner_mask
                    )
                num_adj = self._num_adj[from_pos]
                current_mask = lax.fori_loop(0, num_adj, check_adj_move, current_mask)

                def check_jump_move(j, inner_mask):
                    end_pos = self._jump_adj_matrix[from_pos, j]
                    to_idx = end_pos - 1
                    is_empty = (state.board[to_idx] == 0)

                    from_neighbors_padded = self._adj_matrix[from_pos]
                    to_neighbors_padded = self._adj_matrix[to_idx + 1]
                    from_mask = jnp.arange(self.max_neighbors) < self._num_adj[from_pos]
                    to_mask = jnp.arange(self.max_neighbors) < self._num_adj[to_idx + 1]
                    from_valid = jnp.where(from_mask, from_neighbors_padded, -1)
                    to_valid = jnp.where(to_mask, to_neighbors_padded, -2)

                    mid_pos_arr = jnp.intersect1d(from_valid, to_valid, size=self.max_neighbors, fill_value=0)

                    def process_jump(inner_mask_):
                        mid_pos = mid_pos_arr[0]
                        mid_idx = mid_pos - 1
                        is_goat_in_middle = (state.board[mid_idx] == 1)
                        move_tuple = jnp.array([from_pos, end_pos])
                        action_idx = jnp.where(jnp.all(self._move_action_map == move_tuple, axis=1), size=1, fill_value=-1)[0][0]
                        return lax.cond(
                            (action_idx != -1) & is_empty & is_goat_in_middle,
                            lambda: inner_mask_.at[action_idx].set(True),
                            lambda: inner_mask_
                        )
                    return lax.cond(mid_pos_arr[0] > 0, process_jump, lambda m: m, inner_mask)

                num_jump = self._num_jump_adj[from_pos]
                current_mask = lax.fori_loop(0, num_jump, check_jump_move, current_mask)
                return current_mask

            return lax.fori_loop(0, self.NUM_TIGERS, check_tiger_moves, move_mask)

        is_goat_turn = state.current_player == 0
        is_placement_phase = state.goats_to_place > 0
        placement_mask = lax.cond(is_goat_turn & is_placement_phase, goat_placement_mask, lambda: jnp.zeros(self.placement_actions_count, dtype=jnp.bool_))
        movement_mask = lax.cond(is_goat_turn & ~is_placement_phase, goat_move_mask, lambda: lax.cond(~is_goat_turn, tiger_move_mask, lambda: jnp.zeros(self.move_actions_count, dtype=jnp.bool_)))
        return jnp.concatenate([placement_mask, movement_mask])

    def _are_tigers_blocked(self, board: chex.Array) -> chex.Array:
        temp_state = State(
            current_player=jnp.int32(1),
            observation=jnp.zeros((self.BOARD_POSITIONS * 3 + 4,), dtype=jnp.float32),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=jnp.zeros(self.total_actions, dtype=jnp.bool_),
            _step_count=jnp.int32(0),
            board=board,
            goats_to_place=jnp.int32(0),
            goats_captured=jnp.int32(0)
        )
        mask = self._compute_legal_actions(temp_state)
        return ~jnp.any(mask)

    def _step(self, state: State, action: chex.Array, key: chex.PRNGKey) -> State:
        def perform_action(state, action):
            is_placement = action < self.placement_actions_count

            def place_goat():
                to_idx = action
                newboard = state.board.at[to_idx].set(1)
                newgoats_to_place = state.goats_to_place - 1
                return newboard, newgoats_to_place, state.goats_captured, jnp.array(0.0)

            def move_piece():
                move_idx = action - self.placement_actions_count
                from_pos, to_pos = self._move_action_map[move_idx]
                from_idx, to_idx = from_pos - 1, to_pos - 1
                piece = state.board[from_idx]
                tempboard = state.board.at[from_idx].set(0)
                newboard = tempboard.at[to_idx].set(piece)

                def tiger_jump():
                    from_neighbors_padded = self._adj_matrix[from_pos]
                    to_neighbors_padded = self._adj_matrix[to_pos]
                    from_mask = jnp.arange(self.max_neighbors) < self._num_adj[from_pos]
                    to_mask = jnp.arange(self.max_neighbors) < self._num_adj[to_pos]
                    from_valid = jnp.where(from_mask, from_neighbors_padded, -1)
                    to_valid = jnp.where(to_mask, to_neighbors_padded, -2)

                    mid_pos_arr = jnp.intersect1d(from_valid, to_valid, size=self.max_neighbors, fill_value=0)

                    def capture_goat():
                        mid_idx = mid_pos_arr[0] - 1
                        b = newboard.at[mid_idx].set(0)
                        g_cap = state.goats_captured + 1
                        r = jnp.array(5.0)
                        return b, state.goats_to_place, g_cap, r

                    return lax.cond(
                        mid_pos_arr[0] > 0,
                        capture_goat,
                        lambda: (newboard, state.goats_to_place, state.goats_captured, jnp.array(0.0))
                    )

                finalboard, g_place, g_cap, r = lax.cond(
                    piece == 2,
                    tiger_jump,
                    lambda: (newboard, state.goats_to_place, state.goats_captured, jnp.array(0.0))
                )
                return finalboard, g_place, g_cap, r

            newboard, newgoats_to_place, newgoats_captured, reward = lax.cond(is_placement, place_goat, move_piece)

            new_step_count = state._step_count + 1
            goat_win = self._are_tigers_blocked(newboard)
            tiger_win = newgoats_captured >= self.TIGER_WIN_THRESHOLD
            draw = new_step_count >= self.MAX_TURNS
            terminated = goat_win | tiger_win | draw

            goat_reward = lax.cond(goat_win, lambda: 1.0, lambda: lax.cond(tiger_win, lambda: -1.0, lambda: 0.0))
            tiger_reward = -goat_reward
            
            tiger_reward = tiger_reward + lax.cond(state.current_player == 1, lambda: reward, lambda: 0.0)
            
            rewards = jnp.array([goat_reward, tiger_reward], dtype=jnp.float32)
            
            next_player = 1 - state.current_player
            new_obs = self._make_observation(newboard, next_player, newgoats_to_place, newgoats_captured, new_step_count)
            
            next_state = State(
                current_player=next_player,
                observation=new_obs,
                rewards=rewards,
                terminated=terminated,
                truncated=jnp.bool_(False),
                legal_action_mask=jnp.zeros(self.total_actions, dtype=jnp.bool_),
                _step_count=new_step_count,
                board=newboard,
                goats_to_place=newgoats_to_place,
                goats_captured=newgoats_captured
            )
            
            legal_mask = self._compute_legal_actions(next_state)
            next_state = next_state.replace(legal_action_mask=legal_mask)
            
            return next_state

        return lax.cond(state.terminated, lambda: state, lambda: perform_action(state, action))

    def _make_observation(self, board: chex.Array, player: chex.Array, goats_to_place: chex.Array, 
                          goats_captured: chex.Array, _step_count: chex.Array) -> chex.Array:
        board_one_hot = jax.nn.one_hot(board, 3).flatten()
        state_features = jnp.array([
            player,
            goats_to_place / self.NUM_GOATS,
            goats_captured / self.TIGER_WIN_THRESHOLD,
            _step_count / self.MAX_TURNS
        ], dtype=jnp.float32)
        return jnp.concatenate([board_one_hot, state_features])
    
    def _observe(self, state: State, player_id: chex.Array) -> chex.Array:
        return state.observation
    
    @property
    def id(self) -> str:
        return "aadu_puli_aattam"
    
    @property
    def version(self) -> str:
        return "v0"
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def observation_shape(self):
        return (self.BOARD_POSITIONS * 3 + 4,)
    
    @property  
    def num_actions(self):
        return self.total_actions


def test_single_env():
    """Test single environment initialization and step."""
    print("=" * 60)
    print("Test 1: Single Environment")
    print("=" * 60)
    
    env = AaduPuliAattamJAX()
    key = random.PRNGKey(42)
    
    state = env.init(key)
    
    print(f"✓ Environment initialized")
    print(f"  - Observation shape: {state.observation.shape}")
    print(f"  - Number of actions: {env.num_actions}")
    print(f"  - Current player: {state.current_player}")
    print(f"  - Legal actions available: {jnp.sum(state.legal_action_mask)}")
    print(f"  - Terminated: {state.terminated}")
    
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
    
    key = random.PRNGKey(123)
    keys = random.split(key, batch_size)
    
    print(f"Initializing {batch_size} environments in parallel...")
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    print(f"✓ {batch_size} environments initialized")
    print(f"  - Observation shape: {states.observation.shape}")
    print(f"  - Legal action mask shape: {states.legal_action_mask.shape}")
    print(f"  - Current players shape: {states.current_player.shape}")
    print(f"  - Average legal actions per env: {jnp.mean(jnp.sum(states.legal_action_mask, axis=1)):.1f}")
    
    print(f"\nExecuting vectorized steps...")
    
    def get_first_legal_action(mask):
        legal_actions = jnp.where(mask, size=1, fill_value=0)[0]
        return legal_actions[0]
    
    actions = vmap(get_first_legal_action)(states.legal_action_mask)
    
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
    print("Test 3: Full Game Rollout (8 parallel games, 30 steps)")
    print("=" * 60)
    
    env = AaduPuliAattamJAX()
    batch_size = 8
    num_steps = 30
    
    key = random.PRNGKey(456)
    keys = random.split(key, batch_size)
    
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    print(f"Starting {batch_size} parallel games...")
    
    step_fn = jax.vmap(env.step)
    
    def select_random_legal_action(key, mask):
        legal_indices = jnp.where(mask, size=env.num_actions, fill_value=-1)[0]
        valid_count = jnp.sum(mask)
        random_idx = random.randint(key, (), 0, jnp.maximum(valid_count, 1))
        action = jnp.where(valid_count > 0, legal_indices[random_idx], 0)
        return action
    
    for step in range(num_steps):
        step_keys = random.split(key, batch_size)
        key = random.split(key)[0]
        actions = vmap(select_random_legal_action)(step_keys, states.legal_action_mask)
        
        states = step_fn(states, actions)
        
        terminated_count = jnp.sum(states.terminated)
        if step % 10 == 0 or step == num_steps - 1:
            print(f"  Step {step+1:2d}: {terminated_count}/{batch_size} games finished")
        
        if jnp.all(states.terminated):
            print(f"\n✓ All games finished at step {step+1}")
            break
    
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
    
    print(f"Benchmarking vectorized execution ({batch_size} envs, {num_steps} steps)...")
    keys = random.split(key, batch_size)
    
    start = time.time()
    init_fn = jax.vmap(env.init)
    states = init_fn(keys)
    
    step_fn = jax.vmap(env.step)
    for _ in range(num_steps):
        actions = jnp.zeros(batch_size, dtype=jnp.int32)
        states = step_fn(states, actions)
    
    jax.block_until_ready(states)
    vectorized_time = time.time() - start
    
    print(f"✓ Vectorized execution time: {vectorized_time:.4f}s")
    print(f"  ({batch_size * num_steps / vectorized_time:.1f} env-steps/sec)")
    
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
    import sys
    
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
