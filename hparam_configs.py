"""Named SB3 hyperparameter variants for the iteration-6 PPO/SAC/A2C sweep.

Motivation (see SESSION_CONTEXT_AND_FINDINGS.md, section 9): five ablation
iterations (200k-2M steps, sparse vs. shaped rewards, with and without
warm-start) never produced a policy that *learned* to grasp -- PPO/SAC/A2C
neither collapsed (unlike DDPG/TD3) nor converged on real contact, they
just stayed flat. All prior runs used plain SB3 defaults (see pipeline.py's
`model = algo_class("MlpPolicy", train_env, verbose=0, seed=seed)`, no
learning_rate/ent_coef/policy_kwargs override). This module is the single
source of truth for the hyperparameter variants tried instead, read by
both run_hparam_sweep.py (which runs them) and HYPERPARAMETER_SWEEP.md
(which documents them).

Each variant changes exactly one axis relative to the SB3 defaults, kept
deliberately small (3 variants x 3 algorithms = 9 runs) rather than a full
grid, to fit the established one-day-per-iteration wall-clock budget:

- "larger_net": bigger policy/value network. PPO/A2C default to
  net_arch=[64, 64] (verified via `model.policy_kwargs` on a freshly
  constructed model); SAC already defaults to [256, 256], so its
  "larger_net" variant goes to [400, 400] to still be a real increase.
- "high_entropy": more exploration pressure. PPO/A2C default ent_coef=0.0
  (no entropy bonus at all); SAC defaults to ent_coef="auto" (already
  adaptive) -- its variant overrides that with a fixed, deliberately high
  coefficient rather than tuning the auto target, to keep the "more
  exploration, less exploitation" framing consistent across algorithms.
- "high_lr": ~3x default learning_rate (PPO/SAC 3e-4 -> 1e-3, A2C
  7e-4 -> 2e-3).

The "baseline" (SB3 defaults, no algo_kwargs) is intentionally *not*
repeated here -- it's already covered by iteration 5's phase1_warmstart
runs (results/ablation/phase1_warmstart/<algo>/seed0/), which used exactly
these defaults at the same 400k-step budget. Re-running it would just
burn compute to reproduce numbers already on disk.
"""

HPARAM_VARIANTS = {
    "PPO": {
        "larger_net": {"policy_kwargs": {"net_arch": [256, 256]}},
        "high_entropy": {"ent_coef": 0.02},
        "high_lr": {"learning_rate": 1e-3},
    },
    "SAC": {
        "larger_net": {"policy_kwargs": {"net_arch": [400, 400]}},
        "high_entropy": {"ent_coef": 0.1},
        "high_lr": {"learning_rate": 1e-3},
    },
    "A2C": {
        "larger_net": {"policy_kwargs": {"net_arch": [256, 256]}},
        "high_entropy": {"ent_coef": 0.02},
        "high_lr": {"learning_rate": 2e-3},
    },
}
