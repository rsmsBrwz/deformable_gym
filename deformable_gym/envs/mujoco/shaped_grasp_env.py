"""Dense reward-shaping variant of GraspEnv for deformable-object grasping.

GraspEnv itself stays untouched (sparse task reward only) -- this module adds
a separate subclass so both variants remain independently usable, e.g. for
comparing shaped vs. sparse training. See docs/... session notes and the
externally provided reward-shaping plan for the design rationale.

Reward = w_task * task_reward
       + reward_stability_step (per-step grasped/contact quality)
       + reward_event (drop penalty, acquire bonus, contact-quality progress)
       + reward_terminal_stability (retained ratio, energy penalty, dynamic
         stability -- only nonzero on the terminating step)

Phase 1 (task reward, per-step grasped/contact, drop penalty) is active by
default. Phase 2 (acquire bonus, progress reward) and Phase 3 (terminal
retention/energy/dynamic-stability rewards) default to weight 0.0, so they
are opt-in via constructor kwargs (e.g. ``gym.make(env_id, w_retained=0.2)``)
without any code change -- this is what makes ablations possible.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ...helpers import grasp_metrics as gm
from .grasp_env import GraspEnv


class ShapedGraspEnv(GraspEnv):
    """GraspEnv with dense, configurable reward shaping.

    All weights default to the conservative "Phase 1" setup recommended in
    the reward-shaping plan: unchanged terminal task reward, small per-step
    grasp/contact rewards, and a drop penalty. Phase 2/3 terms are present
    but inert (weight 0.0) until explicitly enabled.
    """

    def __init__(
        self,
        *args,
        w_task: float = 1.0,
        w_grasped: float = 0.01,
        w_contact: float = 0.01,
        contact_force_saturation: float = 5.0,
        w_drop: float = -0.5,
        w_acquire: float = 0.0,
        w_progress: float = 0.0,
        w_retained: float = 0.0,
        w_energy: float = 0.0,
        energy_reference: float = 1.0,
        w_dynamic: float = 0.0,
        dynamic_displacement_reference: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.w_task = w_task
        self.w_grasped = w_grasped
        self.w_contact = w_contact
        self.contact_force_saturation = contact_force_saturation
        self.w_drop = w_drop
        self.w_acquire = w_acquire
        self.w_progress = w_progress
        self.w_retained = w_retained
        self.w_energy = w_energy
        # Rough first calibration from energy_kinetic_before values observed
        # in this project's training runs (typically 0.001-0.2, occasionally
        # higher) -- revisit experimentally rather than trusting this as-is.
        self.energy_reference = energy_reference
        self.w_dynamic = w_dynamic
        # dynamic_max_displacement was consistently ~0.005 across observed
        # stable episodes; 2x that gives a reasonably sensitive scale.
        self.dynamic_displacement_reference = dynamic_displacement_reference

        self._prev_grasped = False
        self._prev_contact_quality = 0.0
        self._ever_grasped_this_episode = False
        self._current_grasped = False
        self._current_contact_quality = 0.0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_grasped = False
        self._prev_contact_quality = 0.0
        self._ever_grasped_this_episode = False
        self._current_grasped = False
        self._current_contact_quality = 0.0
        return obs, info

    def _compute_stability_step_reward(self) -> tuple[float, dict]:
        """Dense per-step reward from contact/grasp state.

        Must run before _compute_task_reward: the latter may trigger
        _pause_simulation (when terminated), which disables constraints and
        steps physics forward -- that would corrupt the contact state this
        method needs to reflect the actual result of the just-applied action.
        """
        contact_ids = gm.hand_object_contact_ids(
            self.model, self.data, self._hand_body_ids, self._object_body_ids
        )
        score = gm.contact_score(self.model, self.data, contact_ids)
        self._current_grasped = gm.binary_grasp_state(score)
        self._current_contact_quality = float(
            min(score.total_normal_force / self.contact_force_saturation, 1.0)
        )

        reward_grasped = self.w_grasped * float(self._current_grasped)
        reward_contact = self.w_contact * self._current_contact_quality
        info = {"reward_grasped": reward_grasped, "reward_contact": reward_contact}
        return reward_grasped + reward_contact, info

    def _compute_event_reward(self) -> tuple[float, dict]:
        """Reward for state transitions: drop, first acquisition, progress.

        Compares self._current_* (set by _compute_stability_step_reward for
        this step) against self._prev_* (from the previous step), then
        advances self._prev_* for the next call.
        """
        drop = self._prev_grasped and not self._current_grasped
        acquire = (
            self._current_grasped
            and not self._prev_grasped
            and not self._ever_grasped_this_episode
        )
        progress = self._current_contact_quality - self._prev_contact_quality

        reward_drop = self.w_drop if drop else 0.0
        reward_acquire = self.w_acquire if acquire else 0.0
        reward_progress = self.w_progress * progress

        if acquire:
            self._ever_grasped_this_episode = True
        self._prev_grasped = self._current_grasped
        self._prev_contact_quality = self._current_contact_quality

        info = {
            "reward_drop": reward_drop,
            "reward_acquire": reward_acquire,
            "reward_progress": reward_progress,
        }
        return reward_drop + reward_acquire + reward_progress, info

    def _compute_task_reward(self, terminated: bool) -> float:
        """Unchanged terminal task reward, reusing GraspEnv._get_reward.

        Note: this is the call that runs _pause_simulation (and thereby
        populates self._episode_grasp_metrics) when terminated -- must run
        after _compute_stability_step_reward, before
        _compute_terminal_stability_reward.
        """
        return float(self._get_reward(terminated))

    def _compute_terminal_stability_reward(self, terminated: bool) -> tuple[float, dict]:
        if not terminated:
            zeros = {"reward_retention": 0.0, "reward_energy": 0.0, "reward_dynamic": 0.0}
            return 0.0, zeros

        metrics = self._episode_grasp_metrics

        retained = metrics.get("retained_ratio", float("nan"))
        reward_retention = self.w_retained * (retained if np.isfinite(retained) else 0.0)

        kinetic_after = metrics.get("energy_kinetic_after", float("nan"))
        if np.isfinite(kinetic_after):
            saturation = min(kinetic_after / self.energy_reference, 1.0)
            reward_energy = -self.w_energy * saturation
        else:
            reward_energy = 0.0

        displacement = metrics.get("dynamic_max_displacement", float("nan"))
        if np.isfinite(displacement):
            closeness = max(0.0, 1.0 - displacement / self.dynamic_displacement_reference)
            reward_dynamic = self.w_dynamic * closeness
        else:
            reward_dynamic = 0.0

        info = {
            "reward_retention": reward_retention,
            "reward_energy": reward_energy,
            "reward_dynamic": reward_dynamic,
        }
        return reward_retention + reward_energy + reward_dynamic, info

    def _compute_total_reward(self, terminated: bool) -> tuple[float, dict]:
        step_reward, step_info = self._compute_stability_step_reward()
        event_reward, event_info = self._compute_event_reward()
        task_reward = self._compute_task_reward(terminated)
        terminal_reward, terminal_info = self._compute_terminal_stability_reward(terminated)

        reward_task_terminal = self.w_task * task_reward
        total = reward_task_terminal + step_reward + event_reward + terminal_reward

        info = {
            "reward_total": total,
            "reward_task_terminal": reward_task_terminal,
            "reward_stability_step": step_reward,
            "reward_event": event_reward,
            "reward_terminal_stability": terminal_reward,
            **step_info,
            **event_info,
            **terminal_info,
        }
        return total, info

    def step(self, action):
        """Re-implements GraspEnv.step to fix a termination-timing issue:
        GraspEnv captures sim_time *before* mj_step, so termination is
        decided on the pre-step time. Kept as-is in GraspEnv (out of scope
        to change there); fixed here since it matters more with a reward
        that depends on precise step-to-step state transitions.
        """
        if self.control_type == "mocap":
            self.mocap.set_ctrl(self.model, self.data, action[:6])
            self.robot.set_ctrl(self.model, self.data, action[6:])
        else:
            self.robot.set_ctrl(self.model, self.data, action)
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        observation = self._get_observation()
        terminated = self._is_terminated(self.data.time)
        truncated = self._is_truncated()

        reward, reward_info = self._compute_total_reward(terminated)
        info = self._get_info()
        info.update(reward_info)
        return observation, reward, terminated, truncated, info
