from __future__ import annotations

from .base_mjenv import BaseMJEnv
import numpy as np

class MoveEnv(BaseMJEnv):
    """
    RL-Umgebung für Bewegungsaufgaben mit einer virtuellen Hand und verformbarem Objekt.
    Die Hand soll das Objekt greifen und gezielt bewegen.
    """
    def __init__(
        self,
        robot_name: str,
        obj_name: str,
        frame_skip: int = 3,
        observable_object_pos: bool = True,
        control_type: str = "mocap",
        max_sim_time: float = 6,
        render_mode: str | None = None,
        mocap_cfg: dict[str, str] | None = None,
        init_frame: str | None = None,
        default_cam_config: dict[str, any] | None = None,
        camera_name: str | None = None,
        camera_id: int | None = None,
    ):
        super().__init__(
            robot_name,
            obj_name,
            frame_skip,
            observable_object_pos,
            control_type,
            max_sim_time,
            render_mode,
            mocap_cfg,
            init_frame,
            default_cam_config,
            camera_name,
            camera_id,
        )
        self.reward_range = (-1, 1)

    def compute_reward(self) -> float:
        # Reward für das Greifen und Bewegen des Objekts (z.B. Annäherung an Zielposition)
        # Platzhalter: Belohnt Annäherung an eine Zielposition
        target_pos = np.array([0.0, 0.0, 0.1])
        obj_pos = self.get_object_position()
        dist = np.linalg.norm(obj_pos - target_pos)
        reward = 1.0 - dist
        return reward

    def step(self, action):
        obs, _, terminated, info = super().step(action)
        reward = self.compute_reward()
        return obs, reward, terminated, info

    def get_object_position(self):
        # Holt die aktuelle Objektposition aus der Simulation
        # Platzhalter: Gibt Nullvektor zurück
        return np.zeros(3)

