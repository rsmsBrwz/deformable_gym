"""Grasp-stability measures for MuJoCo grasp environments.

Provides cheap per-step measures (contact score, binary grasp state) and
more expensive episodic measures (object retained ratio, energy quality,
dynamic stability probe) that can be exposed through an environment's info
dict for logging/analysis. Reward computation is intentionally untouched by
this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from mujoco import MjData, MjModel

from . import mj_utils as mju


@dataclass
class ContactScore:
    n_contacts: int
    total_normal_force: float
    mean_normal_force: float


def _resolve_side_body_id(
    model: MjModel, geom_id: int, flex_id: int
) -> int | None:
    """Resolve which body a contact side belongs to.

    A contact side is either a rigid geom (geom_id >= 0) or a flex/soft-body
    element (flex_id >= 0, geom_id == -1). Flex elements are resolved to the
    body sharing the flex's name, which is how flexcomp objects expose their
    root body in this project's assets (see assets/objects/mjcf/boxes.xml).
    """
    if geom_id is not None and geom_id >= 0:
        return int(model.geom_bodyid[geom_id])
    if flex_id is not None and flex_id >= 0:
        flex_name = mju.id2name(model, flex_id, "flex")
        try:
            return mju.name2id(model, flex_name, "body")
        except ValueError:
            return None
    return None


def hand_object_contact_ids(
    model: MjModel,
    data: MjData,
    hand_body_ids: list[int],
    object_body_ids: list[int],
) -> list[int]:
    """Find indices into ``data.contact`` that connect the hand subtree with
    the object subtree, including flex/soft-body contacts.
    """
    hand_set = set(hand_body_ids)
    object_set = set(object_body_ids)
    contact_ids = []
    for i in range(data.ncon):
        con = data.contact[i]
        body_a = _resolve_side_body_id(model, con.geom1, int(con.flex[0]))
        body_b = _resolve_side_body_id(model, con.geom2, int(con.flex[1]))
        if body_a is None or body_b is None:
            continue
        if (body_a in hand_set and body_b in object_set) or (
            body_b in hand_set and body_a in object_set
        ):
            contact_ids.append(i)
    return contact_ids


def contact_score(
    model: MjModel, data: MjData, contact_ids: list[int]
) -> ContactScore:
    """Aggregate contact count and normal force over the given contacts."""
    if not contact_ids:
        return ContactScore(n_contacts=0, total_normal_force=0.0, mean_normal_force=0.0)
    result = np.zeros(6)
    forces = []
    for i in contact_ids:
        mujoco.mj_contactForce(model, data, i, result)
        forces.append(abs(result[0]))
    return ContactScore(
        n_contacts=len(contact_ids),
        total_normal_force=float(np.sum(forces)),
        mean_normal_force=float(np.mean(forces)),
    )


def binary_grasp_state(
    score: ContactScore, min_contacts: int = 1, min_force: float = 0.02
) -> bool:
    """Whether the object is currently considered held by the hand."""
    return score.n_contacts >= min_contacts and score.total_normal_force >= min_force


def object_retained_ratio(
    model: MjModel,
    data: MjData,
    hand_body_ids: list[int],
    object_part_root_names: list[str],
) -> float:
    """Fraction of the object's named parts that currently have hand contact.

    For a composite object like ``boxes`` (box_left/box_center/box_right),
    this measures how many of the parts are still gripped rather than only
    whether *any* contact exists.
    """
    if not object_part_root_names:
        return float("nan")
    n_gripped = 0
    for part_name in object_part_root_names:
        part_ids = mju.get_body_subtree_ids(model, part_name)
        if hand_object_contact_ids(model, data, hand_body_ids, part_ids):
            n_gripped += 1
    return n_gripped / len(object_part_root_names)


def energy_quality(data: MjData) -> dict:
    """Raw potential/kinetic energy of the whole model.

    Requires ``model.opt.enableflags`` to have ``mjENBL_ENERGY`` set,
    otherwise both values stay at 0. This mixes gravitational and elastic
    (flex) potential energy, so it is a coarse proxy, not an isolated
    deformation-energy measure.
    """
    potential, kinetic = data.energy
    return {"potential": float(potential), "kinetic": float(kinetic)}


def dynamic_stability_probe(
    model: MjModel,
    data: MjData,
    object_root_body_name: str,
    force: tuple[float, float, float] = (1.0, 0.0, 0.0),
    apply_steps: int = 5,
    settle_steps: int = 20,
    settle_speed_threshold: float = 0.01,
) -> dict:
    """Apply a brief external push to the object and measure the response.

    The object's root body (e.g. ``boxes``) is typically a massless
    container; the actual dynamics live in its descendant bodies that carry
    degrees of freedom (flex vertices). The push is applied to all such
    descendant bodies, and the response is measured via the root body's
    ``subtree_com``, consistent with how object height is already judged in
    ``GraspEnv._get_reward``.

    ``settle_step`` counts how many of the ``settle_steps`` steps after the
    push it takes until the object's speed drops below
    ``settle_speed_threshold`` and stays there; it equals ``settle_steps``
    if the object is still moving faster than that at the end of the probe
    window (a cheap proxy for "time to come to rest" -- how quickly the
    deformable object stabilizes after a perturbation).
    """
    object_body_ids = mju.get_body_subtree_ids(model, object_root_body_name)
    dynamic_body_ids = [b for b in object_body_ids if model.body_dofnum[b] > 0]

    start_com = data.body(object_root_body_name).subtree_com.copy()
    max_displacement = 0.0
    force_arr = np.array(force, dtype=np.float64)

    for _ in range(apply_steps):
        for b in dynamic_body_ids:
            data.xfrc_applied[b, :3] = force_arr
        mujoco.mj_step(model, data)
        com = data.body(object_root_body_name).subtree_com
        max_displacement = max(max_displacement, float(np.linalg.norm(com - start_com)))

    for b in dynamic_body_ids:
        data.xfrc_applied[b, :3] = 0.0

    prev_com = data.body(object_root_body_name).subtree_com.copy()
    speeds = []
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        com = data.body(object_root_body_name).subtree_com
        max_displacement = max(max_displacement, float(np.linalg.norm(com - start_com)))
        speeds.append(float(np.linalg.norm(com - prev_com)) / model.opt.timestep)
        prev_com = com.copy()

    final_speed = speeds[-1] if speeds else 0.0
    settle_step = settle_steps
    for i in range(len(speeds)):
        if all(s < settle_speed_threshold for s in speeds[i:]):
            settle_step = i
            break

    return {
        "max_displacement": max_displacement,
        "final_speed": final_speed,
        "settle_step": settle_step,
    }
