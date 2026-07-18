"""Iteration 9 diagnostic: DDPG-specific causes of the frozen-saturated-policy
collapse, now that iteration 7 ruled out action_noise as a sufficient fix.

Motivation (see 02_Sessions/SESSION_CONTEXT_AND_FINDINGS.md section 9, point 3;
03_Experiments/HYPERPARAMETER_SWEEP.md, "Iteration 7"): DDPG reproduces the
same signature as TD3 (action_saturation_mean pinned near 1.0, mean_reward/
n_contacts bit-identical across consecutive eval checkpoints from as early
as the first one) -- but unlike TD3, adding NormalActionNoise does NOT
prevent it; the policy still freezes, just at a different (worse) fixed
point. So the cause has to be something else. Untested so far: actor
network capacity, learning_starts, and learning_rate.

SB3's DDPG constructor applies a single `learning_rate` to both the actor
and critic optimizers (no separate critic_learning_rate kwarg), and does
not expose weight-initialization directly -- so "actor network
initialization" is approximated here via network capacity (policy_kwargs
net_arch) rather than literal init-scheme control.

Variants (one axis changed at a time vs. SB3 defaults, verified via
`DDPG.__init__` signature and a freshly constructed model's `policy.actor.mu`):
- baseline: learning_rate=1e-3, learning_starts=100, net_arch=[400, 300]
  (all SB3 defaults) -- included here (not just referencing prior runs) so
  every variant in this diagnostic is directly comparable under identical
  seed/eval conditions.
- high_learning_starts: learning_starts=10_000 (100x default). More random
  exploration data in the replay buffer before any gradient step, so the
  critic (and thus the actor) isn't fit to a nearly-empty buffer.
- low_lr: learning_rate=1e-4 (10x smaller). Slower actor drift, in case the
  default rate pushes it into tanh saturation before it can discover
  contact-yielding actions.
- larger_net: policy_kwargs=dict(net_arch=[512, 512]). DDPG already
  defaults to [400, 300] (unlike PPO/A2C's [64, 64]), so this variant goes
  beyond that rather than repeating iteration 6's below-default framing.

Scale: 60k steps / eval_freq 10k, same as iteration 7's quick diagnostic,
before committing to a full-scale follow-up for whichever variant (if any)
shows non-frozen and/or contact-positive behavior.

Usage:
    python diagnose_ddpg.py
    python diagnose_ddpg.py --variants high_learning_starts
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

import gymnasium as gym

import deformable_gym  # noqa: F401
import pipeline
from reward_profiles import REWARD_PROFILES

DDPG_VARIANTS = {
    "baseline": {},
    "high_learning_starts": {"learning_starts": 10_000},
    "low_lr": {"learning_rate": 1e-4},
    "larger_net": {"policy_kwargs": {"net_arch": [512, 512]}},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variants", nargs="+", default=list(DDPG_VARIANTS.keys()), choices=list(DDPG_VARIANTS.keys())
    )
    parser.add_argument("--profile", default="phase1_warmstart", choices=list(REWARD_PROFILES.keys()))
    parser.add_argument("--total-timesteps", type=int, default=60_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--idle-timeout-minutes", type=float, default=15)
    parser.add_argument("--run-timeout-minutes", type=float, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--results-dir", default="./results/diagnose_ddpg")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    os.makedirs(cli_args.results_dir, exist_ok=True)
    profile = REWARD_PROFILES[cli_args.profile]

    run_args = SimpleNamespace(
        total_timesteps=cli_args.total_timesteps,
        eval_freq=cli_args.eval_freq,
        n_eval_episodes=cli_args.n_eval_episodes,
        max_episode_steps=cli_args.max_episode_steps,
        idle_timeout_minutes=cli_args.idle_timeout_minutes,
        run_timeout_minutes=cli_args.run_timeout_minutes,
        render=False,
    )

    summary_rows = []
    for variant_name in cli_args.variants:
        algo_kwargs = DDPG_VARIANTS[variant_name]
        for seed in cli_args.seeds:
            run_dir = os.path.join(cli_args.results_dir, "DDPG", variant_name, f"seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
            print(f"=== diagnose: DDPG/{variant_name}/seed{seed} ===")

            final_row = pipeline.run_with_watchdog(
                profile["env_id"],
                "DDPG",
                seed,
                run_args,
                run_dir,
                env_kwargs=profile["kwargs"],
                algo_kwargs=algo_kwargs,
            )
            final_row["algorithm"] = "DDPG"
            final_row["variant"] = variant_name
            final_row["seed"] = seed
            summary_rows.append(final_row)

            summary_path = os.path.join(cli_args.results_dir, "summary.csv")
            fieldnames = sorted({k for row in summary_rows for k in row})
            with open(summary_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

    print(f"Diagnostic summary written to {os.path.join(cli_args.results_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
