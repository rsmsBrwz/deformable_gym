"""Reward-shaping ablation study: sparse vs. Phase 1/1+2/1+2+3 (ShapedGraspEnv).

Runs every (reward profile x SB3 algorithm) combination once, reusing
pipeline.py's subprocess/watchdog/eval-callback machinery for each run (same
robustness guarantees as pipeline.py itself -- a stalled run is killed and
the study continues with the next combination).

Results land under --results-dir/<profile_name>/<algorithm>/seed<seed>/,
i.e. exactly the directory shape analyze_results.py already expects (with
the reward profile taking the "env" slot) -- so the existing, unmodified
analyze_results.py can be pointed at this directory directly:

    python analyze_results.py --results-dir ./results/ablation

A results/ablation/ablation_config.json is written with the exact profile
definitions, git commit, package versions and run parameters, for
reproducibility (see REWARD_SHAPING_ABLATION.md for the write-up).

Usage:
    python run_reward_ablation.py
    python run_reward_ablation.py --algorithms PPO SAC --total-timesteps 5000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from types import SimpleNamespace

import gymnasium as gym
import stable_baselines3

import deformable_gym  # noqa: F401  (registers the Mujoco/PyBullet envs)
import pipeline
from reward_profiles import REWARD_PROFILES


def git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profiles", nargs="+", default=list(REWARD_PROFILES.keys()), choices=list(REWARD_PROFILES.keys())
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(pipeline.ALGORITHMS.keys()),
        choices=list(pipeline.ALGORITHMS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--idle-timeout-minutes", type=float, default=15)
    parser.add_argument("--run-timeout-minutes", type=float, default=240)
    parser.add_argument("--results-dir", default="./results/ablation")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    os.makedirs(cli_args.results_dir, exist_ok=True)

    # SimpleNamespace mirroring the subset of pipeline.parse_args()'s fields
    # that run_with_watchdog()/run_single() actually read.
    run_args = SimpleNamespace(
        total_timesteps=cli_args.total_timesteps,
        eval_freq=cli_args.eval_freq,
        n_eval_episodes=cli_args.n_eval_episodes,
        max_episode_steps=cli_args.max_episode_steps,
        idle_timeout_minutes=cli_args.idle_timeout_minutes,
        run_timeout_minutes=cli_args.run_timeout_minutes,
        render=False,
    )

    ablation_config = {
        "git_commit": git_commit_hash(),
        "run_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__,
        "deformable_gym_version": getattr(deformable_gym, "__version__", "unknown"),
        "cli_args": vars(cli_args),
        "profiles": {name: REWARD_PROFILES[name] for name in cli_args.profiles},
    }
    with open(os.path.join(cli_args.results_dir, "ablation_config.json"), "w") as f:
        json.dump(ablation_config, f, indent=2)

    summary_rows = []
    for profile_name in cli_args.profiles:
        profile = REWARD_PROFILES[profile_name]
        for algo_name in cli_args.algorithms:
            for seed in cli_args.seeds:
                run_dir = os.path.join(
                    cli_args.results_dir, profile_name, algo_name, f"seed{seed}"
                )
                os.makedirs(run_dir, exist_ok=True)

                final_row = pipeline.run_with_watchdog(
                    profile["env_id"],
                    algo_name,
                    seed,
                    run_args,
                    run_dir,
                    env_kwargs=profile["kwargs"],
                )
                # pipeline.run_with_watchdog sets "env" to the raw gym env id
                # (e.g. all three shaped profiles share
                # MjFloatingMiaGraspBoxesShaped-v0). analyze_results.py
                # groups by the "env" column *and* derives it from the
                # directory name -- overwrite it with the profile name so
                # both agree and the three shaped profiles stay
                # distinguishable in the comparison table/plots.
                final_row["gym_env_id"] = final_row.get("env")
                final_row["env"] = profile_name
                final_row["profile"] = profile_name
                summary_rows.append(final_row)

                summary_path = os.path.join(cli_args.results_dir, "summary.csv")
                fieldnames = sorted({k for row in summary_rows for k in row})
                with open(summary_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_rows)

    print(f"Ablation summary written to {os.path.join(cli_args.results_dir, 'summary.csv')}")
    print(f"Ablation config written to {os.path.join(cli_args.results_dir, 'ablation_config.json')}")


if __name__ == "__main__":
    main()
