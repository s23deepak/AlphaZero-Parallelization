"""
Minimal AlphaZero-style search using DeepMind's mctx with JAX/Flax.

This is a lightweight scaffold to run MCTS over an observation space,
producing an improved policy target from a value/policy network.

Notes
- This uses a trivial recurrent function (no learned dynamics). It's enough
  for a smoke test to verify wiring with mctx. Replace with MuZero-style
  representation + dynamics + prediction for a full implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

try:
    import mctx
except ModuleNotFoundError as e:  # pragma: no cover
    raise RuntimeError(
        "mctx is not installed. Please install with: pip install mctx"
    ) from e


class AZNet(nn.Module):
    """Tiny policy/value network.

    Inputs
    - obs: [obs_dim] array

    Outputs
    - policy_logits: [action_dim]
    - value: scalar in [-1, 1]
    """

    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = obs.astype(jnp.float32)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        policy_logits = nn.Dense(self.action_dim)(x)
        value = nn.Dense(1)(x)
        value = jnp.tanh(value)[0]
        return policy_logits, value


@dataclass
class EnvSpec:
    """Environment specification used by the search wrapper.

    - obs_fn: () -> observation for root (1D array)
    - legal_actions_fn: () -> boolean mask over actions for root
    """

    obs_fn: Callable[[], jnp.ndarray]
    legal_actions_fn: Callable[[], jnp.ndarray]
    action_dim: int


def masked_logits(logits: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """Apply -inf to illegal actions to prevent selection during search."""
    neg_inf = jnp.array(-1e9, dtype=logits.dtype)
    return jnp.where(legal_mask, logits, neg_inf)


def make_root_fn(
    apply_fn: Callable,
    env_spec: EnvSpec,
) -> Callable:
    """Create an mctx root function from the network and env spec.

    Returns a function of signature: root_fn(params, rng, unused_obs)
    """

    def root_fn(params, rng, unused_obs):  # mctx API includes obs, we fetch from env instead
        obs = env_spec.obs_fn()
        legal = env_spec.legal_actions_fn()
        logits, value = apply_fn(params, obs)
        logits = masked_logits(logits, legal)
        # Embedding can be the observation itself for a trivial setup
        embedding = obs
        return mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=embedding,
        )

    return root_fn


def make_recurrent_fn(
    apply_fn: Callable,
    env_spec: EnvSpec,
) -> Callable:
    """Create a trivial recurrent function for mctx.

    This function does NOT advance a learned dynamics model; it reuses the
    same observation at each step. It's intended purely for a smoke test.
    """

    def recurrent_fn(params, rng, embedding, action):
        # In a true MuZero impl, you'd compute next embedding via dynamics and
        # predict next priors/value conditioned on (embedding, action).
        # For smoke, reuse root prediction; reward=0.0
        obs = embedding
        legal = env_spec.legal_actions_fn()
        logits, value = apply_fn(params, obs)
        logits = masked_logits(logits, legal)
        return mctx.RecurrentFnOutput(
            reward=jnp.array(0.0, dtype=jnp.float32),
            prior_logits=logits,
            value=value,
            embedding=embedding,
        )

    return recurrent_fn


def run_search(
    params,
    apply_fn: Callable,
    env_spec: EnvSpec,
    rng: jax.Array,
    num_simulations: int = 32,
    temperature: float = 1.0,
):
    """Run mctx search and return policy over actions (improved policy)."""
    root_fn = make_root_fn(apply_fn, env_spec)
    recurrent_fn = make_recurrent_fn(apply_fn, env_spec)

    # Dummy observation placeholder required by API (we ignore inside root_fn)
    dummy_obs = jnp.zeros(1)

    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng,
        root=
        mctx.RootFn(root_fn=root_fn, embed_fn=lambda *_: None, extract_fn=lambda x: x),
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=10,
        gumbel_scale=1.0,
        temperature=temperature,
        # mctx will call root_fn with (params, rng, dummy_obs)
        root_observation=dummy_obs,
    )

    # Convert visit counts to a normalized policy
    pi = policy_output.action_weights  # [action_dim]
    pi = pi / jnp.maximum(jnp.sum(pi), 1e-8)
    return pi, policy_output


def init_network(rng: jax.Array, obs_dim: int, action_dim: int):
    net = AZNet(action_dim=action_dim)
    params = net.init(rng, jnp.zeros((obs_dim,), dtype=jnp.float32))
    return net.apply, params
