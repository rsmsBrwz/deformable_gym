"""Iteration 7 diagnostic: does SB3's action_noise=None default explain the
DDPG/TD3 (and A2C/high_lr) policy collapse seen throughout this study?

Motivation (see SESSION_CONTEXT_AND_FINDINGS.md section 9, point 2/3):
every DDPG/TD3 run across iterations 3-6, plus A2C/high_lr in iteration 6,
shows the same signature -- action_saturation_mean pinned near 1.0 and
mean_reward/n_contacts bit-identical across many consecutive eval
checkpoints, from as early as the very first one (20,000 steps). That
looks like the actor settling into one fixed, saturated action and never
moving again.

DDPG and TD3 are *deterministic* policies -- unlike PPO/SAC/A2C's
stochastic policies, they have no built-in source of behavioral
randomness at all during environment interaction. SB3's constructor
signature exposes an `action_noise` parameter for exactly this reason,
but it defaults to `None`:

    DDPG.__init__(..., action_noise: Optional[ActionNoise] = None, ...)
    TD3.__init__(..., action_noise: Optional[ActionNoise] = None, ...)

Every run in this study (pipeline.py's `run_single`, all iterations)
constructed these algorithms with plain defaults, i.e. **no exploration
noise was ever configured for DDPG/TD3.** After the initial
`learning_starts=100` random-action steps, both algorithms hand control
entirely to the deterministic actor network -- if it settles anywhere
near a saturated tanh output (very plausible: the action space here is
tiny per-step deltas, see below), there is nothing to perturb it back
out. This script tests that hypothesis directly and cheaply (60k steps,
~10-20 min per run rather than a multi-hour ablation) before committing
to a full-scale iteration 7.

Design: for DDPG and TD3, run one "no_noise" (current defaults,
reproducing the collapse pattern) and one "with_noise" variant
(NormalActionNoise sized to 20% of the per-dimension action range,
constant standard deviation for the whole run -- no decay schedule,
kept deliberately simple for this first test) on the same
phase1_warmstart profile used throughout iterations 5-6. If action_noise
is the root cause, "with_noise" should show action_saturation well below
1.0 and should NOT freeze into bit-identical consecutive checkpoints.

Usage:
    python diagnose_action_noise.py
    python diagnose_action_noise.py --algorithms TD3 --total-timesteps 100000
"""

from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace

import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

import gymnasium as gym

import deformable_gym  # noqa: F401
import pipeline
from reward_profiles import REWARD_PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--algorithms", nargs="+", default=["DDPG", "TD3"], choices=["DDPG", "TD3"])
    parser.add_argument("--profile", default="phase1_warmstart", choices=list(REWARD_PROFILES.keys()))
    parser.add_argument("--noise-sigma-fraction", type=float, default=0.2,
                         help="Std-dev of the injected Gaussian action noise, as a "
                         "fraction of each action dimension's (high-low) range.")
    parser.add_argument("--total-timesteps", type=int, default=60_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--idle-timeout-minutes", type=float, default=15)
    parser.add_argument("--run-timeout-minutes", type=float, default=60)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--results-dir", default="./results/diagnose_action_noise")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    os.makedirs(cli_args.results_dir, exist_ok=True)
    profile = REWARD_PROFILES[cli_args.profile]

    # Action-space bounds to size the noise relative to this env's actual
    # (very small, per-step delta) action scale -- see module docstring.
    probe_env = gym.make(profile["env_id"], **profile["kwargs"])
    action_range = probe_env.action_space.high - probe_env.action_space.low
    n_actions = probe_env.action_space.shape[0]
    probe_env.close()
    noise_sigma = cli_args.noise_sigma_fraction * action_range

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
    for algo_name in cli_args.algorithms:
        for variant_name, algo_kwargs in [
            ("no_noise", {}),
            (
                "with_noise",
                {"action_noise": NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_sigma)},
            ),
        ]:
            for seed in cli_args.seeds:
                run_dir = os.path.join(cli_args.results_dir, algo_name, variant_name, f"seed{seed}")
                os.makedirs(run_dir, exist_ok=True)
                print(f"=== diagnose: {algo_name}/{variant_name}/seed{seed} ===")

                final_row = pipeline.run_with_watchdog(
                    profile["env_id"],
                    algo_name,
                    seed,
                    run_args,
                    run_dir,
                    env_kwargs=profile["kwargs"],
                    algo_kwargs=algo_kwargs,
                )
                final_row["algorithm"] = algo_name
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
