"""
Smoke test for the minimal mctx AlphaZero scaffold.

It verifies the end-to-end wiring:
- Initialize tiny network
- Build trivial EnvSpec (fixed observation and legal mask)
- Run mctx search to produce a policy
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp

try:
    import mctx  # noqa: F401
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "mctx not found. Install with: pip install mctx (or pip install git+https://github.com/deepmind/mctx)"
    ) from e

from .agent import EnvSpec, init_network, run_search


def main():
    key = jax.random.PRNGKey(0)
    obs_dim, action_dim = 16, 8

    # Fixed observation and a legal mask that allows half of the actions
    obs = jnp.linspace(-1, 1, obs_dim)
    legal = jnp.arange(action_dim) % 2 == 0  # even actions legal

    def obs_fn():
        return obs

    def legal_fn():
        return legal

    env = EnvSpec(obs_fn=obs_fn, legal_actions_fn=legal_fn, action_dim=action_dim)

    apply_fn, params = init_network(key, obs_dim=obs_dim, action_dim=action_dim)

    pi, out = run_search(
        params,
        apply_fn,
        env,
        rng=key,
        num_simulations=32,
        temperature=1.0,
    )

    # Basic assertions: distribution shape and mass over legal actions only
    assert pi.shape == (action_dim,), f"policy shape {pi.shape} != ({action_dim},)"
    assert jnp.all(pi[~legal] == 0), "Illegal actions received non-zero probability"
    assert jnp.isclose(jnp.sum(pi), 1.0, atol=1e-5), f"policy not normalized: {jnp.sum(pi)}"

    print("mctx AlphaZero smoke test passed.")


if __name__ == "__main__":
    main()
