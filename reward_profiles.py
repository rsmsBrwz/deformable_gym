"""Named reward-shaping configurations for the boxes-grasping ablation study.

Single source of truth for both run_reward_ablation.py (which runs these)
and REWARD_SHAPING_ABLATION.md (which documents them) -- keep in sync by
always reading the weights from here rather than hardcoding them elsewhere.

Each profile maps to a concrete env id (gym.make target) plus the kwargs
used to configure it. "sparse" uses the original, untouched GraspEnv;
the other three progressively enable more of ShapedGraspEnv's reward
components (see deformable_gym/envs/mujoco/shaped_grasp_env.py).
"""

# Warm-start / curriculum initial pose (Iteration 5): move the hand into a
# grasp-ready pose over the left box and partially flex the fingers at
# reset, so the policy starts in sustained contact instead of having to
# discover reach+close from scratch via random exploration.
#
# Calibrated via zero-action rollouts (700 steps, seeds 0-2, see
# REWARD_SHAPING_ABLATION.md, Iteration 5): this pose keeps grasped=True
# for 99.8% of the episode vs. 27% for the default pose. Finger flexion
# alone does NOT work (0.3 rad: 19%, >=0.7 rad: 0% -- curling the fingers
# moves the fingertips away from the box).
#
# Axis-mapping gotcha (measured empirically): the ee slide joints are not
# world-aligned -- ee_X moves the hand along world +y, ee_Y along world
# +x, ee_Z along world -z.
WARM_START_FINGER_FLEX = 0.3
WARM_START_JOINT_TARGETS = {
    joint: WARM_START_FINGER_FLEX
    for joint in ["j_thumb_fle", "j_index_fle", "j_middle_fle", "j_ring_fle", "j_little_fle"]
}
WARM_START_JOINT_TARGETS["ee_X"] = -0.04  # world y: -0.04 (toward the boxes)
WARM_START_JOINT_TARGETS["ee_Y"] = 0.06  # world x: +0.06 (over the left box)

REWARD_PROFILES = {
    "sparse": {
        "env_id": "MjFloatingMiaGraspBoxes-v0",
        "kwargs": {},
        "description": "Baseline: unveraenderter sparse Task-Reward (GraspEnv).",
    },
    "sparse_warmstart": {
        "env_id": "MjFloatingMiaGraspBoxes-v0",
        "kwargs": {"warm_start_joint_targets": WARM_START_JOINT_TARGETS},
        "description": (
            "Wie 'sparse', aber Episode startet mit teilweise geschlossenen "
            "Fingern (Curriculum/Warmstart, Iteration 5) statt der "
            "Standard-Ausgangspose."
        ),
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
    "phase1_warmstart": {
        "env_id": "MjFloatingMiaGraspBoxesShaped-v0",
        "kwargs": {"warm_start_joint_targets": WARM_START_JOINT_TARGETS},
        "description": (
            "Wie 'phase1', aber mit Warmstart-Ausgangspose -- erst dadurch "
            "bekommen die per-step Kontakt-Rewards von Beginn an ein Signal."
        ),
    },
    "phase1_2_3_warmstart": {
        "env_id": "MjFloatingMiaGraspBoxesShaped-v0",
        "kwargs": {
            "w_acquire": 0.5,
            "w_progress": 0.2,
            "w_retained": 0.2,
            "w_energy": 0.1,
            "w_dynamic": 0.1,
            "warm_start_joint_targets": WARM_START_JOINT_TARGETS,
        },
        "description": "Wie 'phase1_2_3', aber mit Warmstart-Ausgangspose.",
    },
}
