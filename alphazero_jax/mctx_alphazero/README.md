# mctx AlphaZero (JAX/Flax) – Minimal Scaffold

This is a tiny AlphaZero-style search wrapper built on DeepMind's `mctx` with JAX/Flax. It’s intended as a wiring smoke test and a starting point for full MuZero-style integration.

What’s included:
- `agent.py`: small policy/value net and `mctx` search glue (root/recurrent fns)
- `smoke_test.py`: quick end-to-end test that returns a legal, normalized policy

Notes:
- The recurrent function does not learn dynamics; it reuses the root prediction. Replace with representation+dynamics+prediction for a full MuZero.
- Illegal actions are masked out in logits.

## Install

```bash
pip install jax jaxlib flax
pip install mctx  # or: pip install git+https://github.com/deepmind/mctx
```

## Run smoke test

```bash
python -m alphazero_jax.mctx_alphazero.smoke_test
```

Expected output:
- "mctx AlphaZero smoke test passed." if everything is wired correctly.

## Next steps
- Integrate your environment’s observation encoder and legal-action mask.
- Replace `recurrent_fn` with learned dynamics (MuZero) or simulate from env.
- Add training loop with policy/value loss and replay buffer.