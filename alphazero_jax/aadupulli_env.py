"""
AaduPulliPGXEnv - A PGX-compatible environment for the traditional South Indian game Aadu Puli Aattam (Goats and Tigers)

This file contains the corrected implementation fixing the reward system bug.
"""

import jax
import jax.numpy as jnp
from pgx.core import Env
from flax.struct import dataclass as struct_dataclass

# --- Constants & Precomputation ---

NUM_GOATS = 15
NUM_TIGERS = 3
TIGER_WIN_THRESHOLD = 10
BOARD_POSITIONS = 23
MAX_TURNS = 200
PLACEMENT_ACTIONS = BOARD_POSITIONS

# Adjacency matrices (1-indexed, 0-padded for JAX compatibility)
ADJ = jnp.array([
    [0, 0, 0, 0], [3, 4, 5, 6], [3, 8, 0, 0], [1, 2, 4, 9], [1, 3, 5, 10],
    [1, 4, 6, 11], [1, 5, 7, 12], [6, 13, 0, 0], [2, 9, 14, 0], [3, 8, 10, 15],
    [4, 9, 11, 16], [5, 10, 12, 17], [6, 11, 13, 18], [7, 12, 14, 0],
    [8, 15, 0, 0], [9, 14, 16, 20], [10, 15, 17, 21], [11, 16, 18, 22],
    [12, 17, 19, 23], [13, 18, 0, 0], [15, 21, 0, 0], [16, 20, 22, 0],
    [17, 21, 23, 0], [18, 22, 0, 0]
], dtype=jnp.int32)

JUMP_ADJ = jnp.array([
    [0, 0, 0, 0], [9, 10, 11, 12], [4, 14, 0, 0], [5, 15, 0, 0], [2, 6, 16, 0],
    [3, 7, 17, 0], [4, 18, 0, 0], [5, 19, 0, 0], [10, 0, 0, 0], [1, 11, 20, 0],
    [1, 8, 12, 21], [1, 9, 13, 22], [1, 10, 23, 0], [11, 0, 0, 0], [2, 16, 0, 0],
    [3, 17, 0, 0], [4, 14, 18, 0], [5, 15, 19, 0], [6, 16, 0, 0], [7, 17, 0, 0],
    [9, 22, 0, 0], [10, 23, 0, 0], [11, 20, 0, 0], [12, 21, 0, 0]
], dtype=jnp.int32)

def _create_move_info():
    """Precomputes detailed information for every possible move."""
    move_info = []
    # Adjacent moves (is_jump=0, mid_pos=0)
    for start_pos in range(1, BOARD_POSITIONS + 1):
        for end_pos in ADJ[start_pos]:
            if end_pos != 0:
                move_info.append([int(start_pos), int(end_pos), 0, 0])

    # Jump moves (is_jump=1, mid_pos=calculated)
    processed_jumps = set()
    for start_pos in range(1, BOARD_POSITIONS + 1):
        for end_pos in JUMP_ADJ[start_pos]:
            end_pos_int = int(end_pos)
            if end_pos_int != 0 and (start_pos, end_pos_int) not in processed_jumps:
                # Find midpoint by checking common neighbors
                mid_pos = 0
                for neighbor1 in ADJ[start_pos]:
                    if neighbor1 != 0:
                        for neighbor2 in ADJ[end_pos_int]:
                            if neighbor2 != 0 and neighbor1 == neighbor2:
                                mid_pos = int(neighbor1)
                                break
                    if mid_pos != 0:
                        break
                move_info.append([start_pos, end_pos_int, 1, mid_pos])
                processed_jumps.add((start_pos, end_pos_int))

    return jnp.array(move_info, dtype=jnp.int32)

MOVE_INFO = _create_move_info()
MOVE_ACTIONS_COUNT = MOVE_INFO.shape[0]
TOTAL_ACTIONS = PLACEMENT_ACTIONS + MOVE_ACTIONS_COUNT

@struct_dataclass
class State:
    """State dataclass for the Aadu Puli Aattam environment."""
    current_player: jnp.ndarray
    observation: jnp.ndarray
    rewards: jnp.ndarray
    terminated: jnp.ndarray
    truncated: jnp.ndarray
    legal_action_mask: jnp.ndarray
    _step_count: jnp.ndarray
    board: jnp.ndarray
    goats_to_place: jnp.ndarray
    goats_captured: jnp.ndarray
    turn_count: jnp.ndarray


class AaduPulliPGXEnv(Env):
    """Aadu Puli Aattam (Goats and Tigers) game environment."""
    version = "v0"
    num_players = 2

    def _init(self, key: jax.random.PRNGKey) -> State:
        """Initializes the game state."""
        board = jnp.zeros(BOARD_POSITIONS, dtype=jnp.int32)
        board = board.at[jnp.array([0, 3, 4])].set(2)  # Initial tiger positions

        state = State(
            current_player=jnp.int32(0),
            board=board,
            goats_to_place=jnp.int32(NUM_GOATS),
            goats_captured=jnp.int32(0),
            turn_count=jnp.int32(0),
            terminated=jnp.bool_(False),
            truncated=jnp.bool_(False),
            legal_action_mask=jnp.zeros(TOTAL_ACTIONS, dtype=jnp.bool_),
            _step_count=jnp.int32(0),
            rewards=jnp.zeros(2, dtype=jnp.float32),
            observation=jnp.zeros(BOARD_POSITIONS + 3, dtype=jnp.int32)
        )
        return state.replace(legal_action_mask=self._legal_action_mask(state))

    def _step(self, state: State, action: jnp.ndarray, key: jax.random.PRNGKey) -> State:
        """Takes a step in the environment."""

        def _handle_placement(state, action):
            new_board = state.board.at[action].set(1)
            return state.replace(
                board=new_board,
                goats_to_place=state.goats_to_place - 1,
            )

        def _handle_movement(state, action):
            move_idx = action - PLACEMENT_ACTIONS
            from_pos, to_pos, is_jump, mid_pos = MOVE_INFO[move_idx]
            from_idx, to_idx = from_pos - 1, to_pos - 1

            piece = state.board[from_idx]
            new_board = state.board.at[from_idx].set(0).at[to_idx].set(piece)

            # Handle goat capture during a tiger jump
            goats_captured = jax.lax.cond(
                (piece == 2) & (is_jump == 1),
                lambda: state.goats_captured + 1,
                lambda: state.goats_captured
            )
            new_board = jax.lax.cond(
                (piece == 2) & (is_jump == 1),
                lambda: new_board.at[mid_pos - 1].set(0),
                lambda: new_board
            )
            return state.replace(board=new_board, goats_captured=goats_captured)

        # Update board based on action type
        intermediate_state = jax.lax.cond(
            action < PLACEMENT_ACTIONS,
            _handle_placement,
            _handle_movement,
            state, action
        )

        # Determine win/loss conditions based on the intermediate state board and goats captured
        t_win = intermediate_state.goats_captured >= TIGER_WIN_THRESHOLD
        draw = intermediate_state.turn_count >= MAX_TURNS

        # Calculate the *next* player and next turn count
        next_player = 1 - intermediate_state.current_player
        next_turn_count = intermediate_state.turn_count + 1

        # Create a temporary state to get the legal actions for the NEXT player on the NEW board
        temp_state_for_legal_mask = intermediate_state.replace(
            current_player=next_player
        )
        next_legal_mask = self._legal_action_mask(temp_state_for_legal_mask)

        # Check for Goat Win (Tigers Blocked) - This happens if it's the tiger's turn next AND they have no legal moves
        is_next_player_tiger = next_player == 1
        next_player_has_no_legal_moves = ~jnp.any(next_legal_mask)
        g_win = is_next_player_tiger & next_player_has_no_legal_moves

        # Determine final terminated status and rewards based on the determined win/loss conditions
        terminated = t_win | draw | g_win

        # FIXED: Calculate reward based on termination condition
        # When tiger wins: Goat (player 0) gets -1, Tiger (player 1) gets +1
        # When goat wins: Goat (player 0) gets +1, Tiger (player 1) gets -1
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

        final_state = intermediate_state.replace(
            current_player=next_player,
            turn_count=next_turn_count,
            terminated=terminated,
            rewards=reward,
            legal_action_mask=next_legal_mask
        )

        return final_state

    def _observe(self, state: State, player_id: jnp.ndarray) -> jnp.ndarray:
        """Returns the observation for the specified player."""
        return jnp.concatenate([
            state.board,
            jnp.array([state.current_player], dtype=jnp.int32),
            jnp.array([state.goats_to_place], dtype=jnp.int32),
            jnp.array([state.goats_captured], dtype=jnp.int32)
        ])

    def _legal_action_mask(self, state: State) -> jnp.ndarray:
        """Computes a boolean mask of legal actions."""

        # Placement Phase Legal Actions
        is_placement_phase = (state.current_player == 0) & (state.goats_to_place > 0)
        can_place = state.board == 0
        placement_mask = is_placement_phase & can_place

        # Movement Phase Legal Actions
        def is_move_legal(move_info):
            from_pos, to_pos, is_jump, mid_pos = move_info
            from_idx, to_idx = from_pos - 1, to_pos - 1

            is_dest_empty = state.board[to_idx] == 0

            # Goat move logic
            is_goat_move = (state.board[from_idx] == 1) & (is_jump == 0)

            # Tiger move logic
            is_tiger_adj_move = (state.board[from_idx] == 2) & (is_jump == 0)
            is_tiger_jump_move = (state.board[from_idx] == 2) & (is_jump == 1) & (state.board[mid_pos - 1] == 1)
            is_tiger_move = is_tiger_adj_move | is_tiger_jump_move

            is_goat_turn = state.current_player == 0
            is_tiger_turn = state.current_player == 1
            is_legal = (is_goat_turn & is_goat_move) | (is_tiger_turn & is_tiger_move)

            return is_legal & is_dest_empty

        is_move_phase = ~is_placement_phase
        move_mask = is_move_phase & jax.vmap(is_move_legal)(MOVE_INFO)

        return jnp.concatenate([placement_mask, move_mask])

    @property
    def id(self) -> str:
        return "aadu_puli_aattam"

    @property
    def num_actions(self) -> int:
        return TOTAL_ACTIONS