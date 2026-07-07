"""Named reward-shaping configurations for the boxes-grasping ablation study.

Single source of truth for both run_reward_ablation.py (which runs these)
and REWARD_SHAPING_ABLATION.md (which documents them) -- keep in sync by
always reading the weights from here rather than hardcoding them elsewhere.

Each profile maps to a concrete env id (gym.make target) plus the kwargs
used to configure it. "sparse" uses the original, untouched GraspEnv;
the other three progressively enable more of ShapedGraspEnv's reward
components (see deformable_gym/envs/mujoco/shaped_grasp_env.py).
"""

REWARD_PROFILES = {
    "sparse": {
        "env_id": "MjFloatingMiaGraspBoxes-v0",
        "kwargs": {},
        "description": "Baseline: unveraenderter sparse Task-Reward (GraspEnv).",
    },
    "phase1": {
        "env_id": "MjFloatingMiaGraspBoxesShaped-v0",
        "kwargs": {},
        "description": (
            "Phase 1: Task-Reward + per-step grasped/Kontaktqualitaet + "
            "Drop-Penalty (ShapedGraspEnv-Defaults, Phase 2/3 aus)."
        ),
    },
    "phase1_2": {
        "env_id": "MjFloatingMiaGraspBoxesShaped-v0",
        "kwargs": {"w_acquire": 0.5, "w_progress": 0.2},
        "description": "Phase 1 + Acquire-Bonus + Kontakt-Progress-Reward.",
    },
    "phase1_2_3": {
        "env_id": "MjFloatingMiaGraspBoxesShaped-v0",
        "kwargs": {
            "w_acquire": 0.5,
            "w_progress": 0.2,
            "w_retained": 0.2,
            "w_energy": 0.1,
            "w_dynamic": 0.1,
        },
        "description": (
            "Phase 1 + 2 + terminale Retention-/Energie-/"
            "Dynamic-Stability-Rewards."
        ),
    },
}
