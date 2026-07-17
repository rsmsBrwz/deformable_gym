"""Iteration 6: PPO/SAC/A2C hyperparameter sweep on the phase1_warmstart profile.

Reuses pipeline.py's subprocess/watchdog/eval-callback machinery (same
robustness guarantees as pipeline.py and run_reward_ablation.py) plus
run_reward_ablation.py's REWARD_PROFILES for the env/reward configuration.
Only the SB3 algorithm hyperparameters vary across runs -- see
hparam_configs.py for the variants and why they were chosen, and
HYPERPARAMETER_SWEEP.md for the write-up.

Results land under --results-dir/<algo>/<variant>/seed<seed>/, with a
matching summary.csv (columns: algorithm, variant, ...).

Usage:
    python run_hparam_sweep.py
    python run_hparam_sweep.py --algorithms PPO --variants larger_net
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
from hparam_configs import HPARAM_VARIANTS
from reward_profiles import REWARD_PROFILES


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--profile",
        default="phase1_warmstart",
        choices=list(REWARD_PROFILES.keys()),
        help="Reward profile (env + env_kwargs) to hold fixed across the sweep.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(HPARAM_VARIANTS.keys()),
        choices=list(HPARAM_VARIANTS.keys()),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Variant names to run (default: all variants defined for each "
        "algorithm in hparam_configs.py).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--total-timesteps", type=int, default=400_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument("--max-episode-steps", type=int, default=800)
    parser.add_argument("--idle-timeout-minutes", type=float, default=15)
    parser.add_argument("--run-timeout-minutes", type=float, default=240)
    parser.add_argument("--results-dir", default="./results/hparam_sweep")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    os.makedirs(cli_args.results_dir, exist_ok=True)

    run_args = SimpleNamespace(
        total_timesteps=cli_args.total_timesteps,
        eval_freq=cli_args.eval_freq,
        n_eval_episodes=cli_args.n_eval_episodes,
        max_episode_steps=cli_args.max_episode_steps,
        idle_timeout_minutes=cli_args.idle_timeout_minutes,
        run_timeout_minutes=cli_args.run_timeout_minutes,
        render=False,
    )

    profile = REWARD_PROFILES[cli_args.profile]

    sweep_config = {
        "git_commit": git_commit_hash(),
        "run_time": datetime.now().isoformat(),
        "python_version": sys.version,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__,
        "deformable_gym_version": getattr(deformable_gym, "__version__", "unknown"),
        "cli_args": vars(cli_args),
        "profile": profile,
        "hparam_variants": {
            algo: {
                v: HPARAM_VARIANTS[algo][v]
                for v in (cli_args.variants or HPARAM_VARIANTS[algo])
            }
            for algo in cli_args.algorithms
        },
    }
    with open(os.path.join(cli_args.results_dir, "sweep_config.json"), "w") as f:
        json.dump(sweep_config, f, indent=2)

    # Load any pre-existing summary.csv rows first -- this script is meant to
    # be re-invoked against the same --results-dir to add more seeds/variants
    # (see the iteration-6b follow-up in HYPERPARAMETER_SWEEP.md). Without
    # this, a second invocation's fresh, empty summary_rows would overwrite
    # the file below and silently discard every row from earlier invocations
    # (the underlying per-run directories/checkpoints are untouched either
    # way, only this aggregate CSV was at risk).
    summary_path = os.path.join(cli_args.results_dir, "summary.csv")
    summary_rows = []
    existing_keys = set()
    if os.path.exists(summary_path):
        with open(summary_path, newline="") as f:
            for row in csv.DictReader(f):
                summary_rows.append(row)
                existing_keys.add((row.get("algorithm"), row.get("variant"), row.get("seed")))

    for algo_name in cli_args.algorithms:
        variant_names = cli_args.variants or list(HPARAM_VARIANTS[algo_name])
        for variant_name in variant_names:
            algo_kwargs = HPARAM_VARIANTS[algo_name][variant_name]
            for seed in cli_args.seeds:
                run_dir = os.path.join(
                    cli_args.results_dir, algo_name, variant_name, f"seed{seed}"
                )
                os.makedirs(run_dir, exist_ok=True)

                final_row = pipeline.run_with_watchdog(
                    profile["env_id"],
                    algo_name,
                    seed,
                    run_args,
                    run_dir,
                    env_kwargs=profile["kwargs"],
                    algo_kwargs=algo_kwargs,
                )
                final_row["gym_env_id"] = final_row.get("env")
                final_row["env"] = f"{algo_name}_{variant_name}"
                final_row["profile"] = cli_args.profile
                final_row["variant"] = variant_name
                key = (algo_name, variant_name, str(seed))
                if key in existing_keys:
                    # Re-running an already-recorded combo (e.g. re-invoked
                    # with overlapping seeds) -- replace its row instead of
                    # duplicating it.
                    summary_rows = [
                        r
                        for r in summary_rows
                        if (r.get("algorithm"), r.get("variant"), r.get("seed")) != key
                    ]
                existing_keys.add(key)
                summary_rows.append(final_row)

                fieldnames = sorted({k for row in summary_rows for k in row})
                with open(summary_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_rows)

    print(f"Sweep summary written to {os.path.join(cli_args.results_dir, 'summary.csv')}")
    print(f"Sweep config written to {os.path.join(cli_args.results_dir, 'sweep_config.json')}")


if __name__ == "__main__":
    main()
