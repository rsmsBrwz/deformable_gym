"""Smoke-test training/eval pipeline for deformable_gym Mujoco grasp environments.

Runs a small grid of Stable-Baselines3 algorithms over one or more grasp
environments and saves both standard SB3 training metrics and custom
grasp-stability measures for later analysis. Defaults are intentionally
small (a few thousand timesteps) so a full run finishes in minutes on a
local machine -- this validates that the training loop and the
(previously untested) environment work together, it does not aim to
produce a well-trained policy.

Usage:
    python pipeline.py --algorithms PPO SAC --seeds 0
    python pipeline.py --envs MjFloatingMiaGraspBoxes-v0 --total-timesteps 20000
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import stable_baselines3
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

import deformable_gym  # noqa: F401  (registers the Mujoco/PyBullet envs)

ALGORITHMS = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C, "DDPG": DDPG}

# Grasp-stability measures exposed by GraspEnv's info dict (see
# deformable_gym/envs/mujoco/grasp_env.py and helpers/grasp_metrics.py).
# Always present (NaN until computed), so they are safe to use as SB3
# Monitor info_keywords and in the custom eval callback below.
GRASP_INFO_KEYS = (
    "n_contacts",
    "total_normal_force",
    "grasped",
    "retained_ratio",
    "energy_potential_before",
    "energy_kinetic_before",
    "energy_potential_after",
    "energy_kinetic_after",
    "dynamic_max_displacement",
    "dynamic_final_speed",
    "dynamic_settle_step",
    # Not from GraspEnv -- injected by NanSafeWrapper below. True iff this
    # episode ended via truncation (non-finite obs or the step-count
    # backstop) rather than GraspEnv's normal time-based termination, i.e.
    # the physics simulation went unstable instead of the episode just
    # running its course.
    "sim_unstable",
)

# Reward-component breakdown, only present in ShapedGraspEnv's info dict
# (see deformable_gym/envs/mujoco/shaped_grasp_env.py). Tracked separately
# from GRASP_INFO_KEYS (rather than merged into it) because GRASP_INFO_KEYS
# also feeds SB3's Monitor(info_keywords=...), which asserts every key is
# present -- that would break runs using the plain sparse GraspEnv, which
# never has these keys. GraspMetricsEvalCallback.run_eval() already reads
# keys via info.get(k, np.nan), so missing keys here just show up as NaN.
REWARD_BREAKDOWN_KEYS = (
    "reward_total",
    "reward_task_terminal",
    "reward_stability_step",
    "reward_event",
    "reward_terminal_stability",
    "reward_grasped",
    "reward_contact",
    "reward_drop",
    "reward_acquire",
    "reward_progress",
    "reward_retention",
    "reward_energy",
    "reward_dynamic",
)


class NanSafeWrapper(gym.Wrapper):
    """Guards against MuJoCo soft-body simulation instability.

    Extreme actions can destabilize the deformable-object simulation (NaN/Inf
    in qacc); once that happens the environment's own simulation-time based
    termination can stop advancing reliably, hanging the episode forever.
    This wrapper truncates the episode as soon as a non-finite observation
    is seen, and additionally enforces a hard step-count cap as a backstop
    independent of the environment's internal sim time.
    """

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._steps = 0

    def reset(self, **kwargs):
        self._steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        if not np.all(np.isfinite(obs)) or self._steps >= self.max_episode_steps:
            truncated = True
        info["sim_unstable"] = bool(truncated)
        return obs, reward, terminated, truncated, info


def make_env(
    env_id: str,
    seed: int,
    max_episode_steps: int,
    render: bool = False,
    env_kwargs: dict | None = None,
) -> gym.Env:
    env = gym.make(env_id, render_mode="human" if render else None, **(env_kwargs or {}))
    env = NanSafeWrapper(env, max_episode_steps=max_episode_steps)
    env.reset(seed=seed)
    return env


class GraspMetricsEvalCallback(BaseCallback):
    """Periodically evaluates the current policy deterministically and logs
    reward, success and grasp-stability measures to both the SB3 logger
    (visible in progress.csv/tensorboard) and a dedicated grasp_stability.csv.

    Also tracks the best-scoring checkpoint seen so far (see _score) and
    keeps it saved separately as best_model.zip -- motivated by an ablation
    run where a policy (sparse/A2C) briefly learned to hold the object
    around the middle of training and then regressed by the end, so
    reporting only the final checkpoint would have missed the best behavior
    actually found during training.
    """

    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int,
        n_eval_episodes: int,
        csv_path: str,
        checkpoint_dir: str,
        extra_info_keys: tuple = (),
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.csv_path = csv_path
        self.checkpoint_dir = checkpoint_dir
        # Keys read via info.get(k, np.nan) below, so envs that don't
        # produce a given key (e.g. plain GraspEnv vs. ShapedGraspEnv's
        # reward_* keys) simply log NaN for it instead of erroring.
        self.tracked_keys = GRASP_INFO_KEYS + tuple(extra_info_keys)
        self._csv_initialized = False
        self.best_score = -np.inf
        self.best_step = None
        self.best_row = None

    @staticmethod
    def score(row: dict) -> float:
        """Ranks checkpoints by mean_reward, but heavily penalizes
        instability. This matters because a checkpoint can show a
        deceptively good-looking mean_reward (e.g. 0.0, higher than a
        genuine failure's -1.0) purely because episodes got truncated by
        NanSafeWrapper before the reward-defining height check ever ran --
        not because the policy is actually doing better. A 2.0-point
        penalty per unit of sim_unstable_mean dominates the [-1, 1] reward
        range, so any checkpoint touched by instability always ranks below
        any fully stable one, regardless of its raw reward.
        """
        return row.get("mean_reward", -np.inf) - 2.0 * row.get("sim_unstable_mean", 0.0)

    def _init_csv(self, fieldnames: list[str]) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        self._csv_initialized = True

    def run_eval(self) -> dict:
        episode_rewards = []
        episode_lengths = []
        episode_success = []
        episode_action_saturation = []
        metric_values = {k: [] for k in self.tracked_keys}

        # Bounds of the action space, used below to detect how often the
        # policy pushes actions all the way to their limits ("saturation") --
        # a common symptom of a policy that has collapsed onto a few extreme
        # actions instead of exercising fine control.
        action_low = np.asarray(self.eval_env.action_space.low, dtype=np.float64)
        action_high = np.asarray(self.eval_env.action_space.high, dtype=np.float64)
        action_range = action_high - action_low
        action_range = np.where(np.isfinite(action_range), action_range, np.nan)

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            terminated = truncated = False
            total_reward = 0.0
            length = 0
            last_info = info
            saturations = []
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                length += 1
                last_info = info
                near_bound = np.abs(action - action_low) < 0.05 * action_range
                near_bound |= np.abs(action_high - action) < 0.05 * action_range
                saturations.append(float(np.nanmean(near_bound)))
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            episode_success.append(bool(total_reward > 0))
            episode_action_saturation.append(
                float(np.mean(saturations)) if saturations else np.nan
            )
            for k in self.tracked_keys:
                v = last_info.get(k, np.nan)
                metric_values[k].append(np.nan if v is None else float(v))

        row = {
            "num_timesteps": self.num_timesteps,
            "mean_reward": float(np.mean(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "success_rate": float(np.mean(episode_success)),
            "action_saturation_mean": float(np.nanmean(episode_action_saturation)),
        }
        for k in self.tracked_keys:
            row[f"{k}_mean"] = float(np.nanmean(metric_values[k]))
        return row

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        row = self.run_eval()
        if not self._csv_initialized:
            self._init_csv(list(row.keys()))
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=list(row.keys())).writerow(row)

        for k, v in row.items():
            if k != "num_timesteps":
                self.logger.record(f"grasp/{k}", v)
        self.logger.dump(self.num_timesteps)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model.save(
            os.path.join(self.checkpoint_dir, f"policy_step{self.num_timesteps}")
        )

        current_score = self.score(row)
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_step = self.num_timesteps
            self.best_row = row
            self.model.save(os.path.join(self.checkpoint_dir, "best_model"))
            with open(os.path.join(self.checkpoint_dir, "best_checkpoint.json"), "w") as f:
                json.dump(
                    {
                        "best_step": self.best_step,
                        "score": self.best_score,
                        "score_formula": "mean_reward - 2.0 * sim_unstable_mean",
                        "metrics": self.best_row,
                    },
                    f,
                    indent=2,
                )
        return True


class HeartbeatCallback(BaseCallback):
    """Reports periodic progress to the parent process via a queue.

    A single native mj_step call can, in rare cases, hang indefinitely once
    the physics state becomes degenerate (e.g. a non-finite position blowing
    up broad-phase collision detection) -- this happened during an overnight
    run and is not something a post-step Python check (like NanSafeWrapper)
    can catch, since the process never returns from that call. The parent's
    watchdog (see run_with_watchdog) uses the absence of heartbeats to
    detect and kill such a stalled run instead of blocking forever.
    """

    def __init__(self, heartbeat_queue: mp.Queue, every_n_steps: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.heartbeat_queue = heartbeat_queue
        self.every_n_steps = every_n_steps

    def _on_step(self) -> bool:
        if self.n_calls % self.every_n_steps == 0:
            self.heartbeat_queue.put(("heartbeat", self.num_timesteps))
        return True


def run_single(
    env_id: str,
    algo_name: str,
    seed: int,
    args: argparse.Namespace,
    run_dir: str,
    result_queue: mp.Queue,
    env_kwargs: dict | None = None,
) -> None:
    """Train+evaluate one (env, algorithm, seed) combination.

    Runs in its own subprocess (see run_with_watchdog) so a stalled run can
    be killed from the outside without affecting the rest of the grid.
    Puts ("heartbeat", num_timesteps) messages periodically and a final
    ("result", row) message via result_queue.

    env_kwargs (e.g. ShapedGraspEnv reward weights) are passed through to
    gym.make() for both the training and eval env -- see
    reward_profiles.py / run_reward_ablation.py for how this is used.
    """
    algo_class = ALGORITHMS[algo_name]
    print(f"=== {env_id} / {algo_name} / seed {seed} ===")

    train_env = make_env(env_id, seed, args.max_episode_steps, args.render, env_kwargs)
    train_env = Monitor(
        train_env,
        filename=os.path.join(run_dir, "monitor.csv"),
        info_keywords=GRASP_INFO_KEYS,
    )
    eval_env = make_env(env_id, seed + 999, args.max_episode_steps, args.render, env_kwargs)

    model = algo_class("MlpPolicy", train_env, verbose=0, seed=seed)
    model.set_logger(configure(run_dir, ["stdout", "csv", "tensorboard"]))

    eval_callback = GraspMetricsEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        csv_path=os.path.join(run_dir, "grasp_stability.csv"),
        checkpoint_dir=os.path.join(run_dir, "checkpoints"),
        extra_info_keys=REWARD_BREAKDOWN_KEYS,
    )
    heartbeat_callback = HeartbeatCallback(result_queue)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[heartbeat_callback, eval_callback],
    )
    model.save(os.path.join(run_dir, "final_model"))

    final_row = eval_callback.run_eval()
    final_row.update(
        {"env": env_id, "algorithm": algo_name, "seed": seed, "timed_out": False}
    )
    # The final checkpoint isn't necessarily the best one seen during
    # training (a policy can regress after finding good behavior, see
    # GraspMetricsEvalCallback.score docstring) -- report the best
    # checkpoint's own eval alongside the final one so both are visible.
    if eval_callback.best_row is not None:
        final_row["best_step"] = eval_callback.best_step
        for k, v in eval_callback.best_row.items():
            if k != "num_timesteps":
                final_row[f"best_{k}"] = v

    train_env.close()
    eval_env.close()

    result_queue.put(("result", final_row))


def run_with_watchdog(
    env_id: str,
    algo_name: str,
    seed: int,
    args: argparse.Namespace,
    run_dir: str,
    env_kwargs: dict | None = None,
) -> dict:
    """Run run_single in a subprocess, killing it if it stalls or runs too long.

    Two independent limits apply: --idle-timeout-minutes bounds the gap
    between two heartbeats (catches a hung/stalled run quickly), and
    --run-timeout-minutes bounds the total wall time of one run (a safety
    net regardless of heartbeats).
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=run_single,
        args=(env_id, algo_name, seed, args, run_dir, result_queue, env_kwargs),
    )
    process.start()

    idle_timeout = args.idle_timeout_minutes * 60
    deadline = time.monotonic() + args.run_timeout_minutes * 60
    final_row = None

    while True:
        wait = max(0.0, min(idle_timeout, deadline - time.monotonic()))
        try:
            kind, payload = result_queue.get(timeout=wait)
        except queue_mod.Empty:
            print(
                f"    [WATCHDOG] {env_id}/{algo_name}/seed{seed}: no progress for "
                f"{args.idle_timeout_minutes} min or {args.run_timeout_minutes} min "
                "hard cap reached -- killing this run."
            )
            break
        if kind == "heartbeat":
            if time.monotonic() > deadline:
                print(
                    f"    [WATCHDOG] {env_id}/{algo_name}/seed{seed}: exceeded "
                    f"{args.run_timeout_minutes} min hard cap -- killing this run."
                )
                break
            continue
        final_row = payload
        break

    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()

    if final_row is None:
        final_row = {
            "env": env_id,
            "algorithm": algo_name,
            "seed": seed,
            "timed_out": True,
        }
        # A killed run may still have found a good checkpoint before it
        # stalled/collapsed -- recover it from disk if present, so a timeout
        # doesn't erase evidence of earlier good behavior.
        best_checkpoint_path = os.path.join(run_dir, "checkpoints", "best_checkpoint.json")
        if os.path.exists(best_checkpoint_path):
            with open(best_checkpoint_path) as f:
                best_checkpoint = json.load(f)
            final_row["best_step"] = best_checkpoint["best_step"]
            for k, v in best_checkpoint["metrics"].items():
                if k != "num_timesteps":
                    final_row[f"best_{k}"] = v
    return final_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=["MjFloatingMiaGraspBoxes-v0"],
        help="Gymnasium env ids to run (default: boxes env only).",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=list(ALGORITHMS.keys()),
        choices=list(ALGORITHMS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000,
        help="Per run. Kept small by default for a quick local smoke test.",
    )
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--n-eval-episodes", type=int, default=3)
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=800,
        help="Hard safety cap (roughly 2x a normal episode) in case the "
        "soft-body simulation destabilizes and never reaches its own "
        "sim-time based termination.",
    )
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument(
        "--idle-timeout-minutes",
        type=float,
        default=15,
        help="Kill a run if no heartbeat/progress is seen for this long "
        "(catches a stalled/hung run, e.g. a single mj_step call that never "
        "returns after the physics state goes non-finite).",
    )
    parser.add_argument(
        "--run-timeout-minutes",
        type=float,
        default=240,
        help="Absolute wall-clock cap per (env, algorithm, seed) run, "
        "regardless of heartbeats.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render training/eval episodes (opens a viewer window per run; "
        "for debugging only, not recommended for the full grid).",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    os.makedirs(args.results_dir, exist_ok=True)
    summary_rows = []

    run_metadata = {
        "python_version": sys.version,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gym.__version__,
        "deformable_gym_version": getattr(deformable_gym, "__version__", "unknown"),
        "run_time": datetime.now().isoformat(),
        "args": vars(args),
    }
    with open(os.path.join(args.results_dir, "run_metadata.json"), "w") as f:
        json.dump(run_metadata, f, indent=2)

    for env_id in args.envs:
        for algo_name in args.algorithms:
            for seed in args.seeds:
                run_dir = os.path.join(args.results_dir, env_id, algo_name, f"seed{seed}")
                os.makedirs(run_dir, exist_ok=True)

                final_row = run_with_watchdog(env_id, algo_name, seed, args, run_dir)
                summary_rows.append(final_row)

                # Write the summary after every run, not just at the end, so
                # a later stall/crash doesn't lose already-finished results.
                summary_path = os.path.join(args.results_dir, "summary.csv")
                fieldnames = sorted({k for row in summary_rows for k in row})
                with open(summary_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(summary_rows)

    if summary_rows:
        print(f"Summary written to {os.path.join(args.results_dir, 'summary.csv')}")


if __name__ == "__main__":
    run(parse_args())
