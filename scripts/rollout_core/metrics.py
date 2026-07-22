"""Physical task milestone evaluation for rollout episodes."""

from typing import TypedDict

import numpy as np

from constant import (
    CUBE_LIFT_MIN_DELTA,
    GRASP_STABLE_STEPS,
    LOWERING_STABLE_STEPS,
    PLACEMENT_MAX_SPEED,
    PLACEMENT_STABLE_STEPS,
    PLACEMENT_Z_TOL,
    RELEASE_STABLE_STEPS,
    RETREAT_MIN_HEIGHT,
    RETREAT_STABLE_STEPS,
    TRAY_INNER_XY_TOL,
    TRAY_PLACE_TOL,
)
from expert import PickPlaceController


class TaskMetrics(TypedDict):
    grasped: bool
    lifted: bool
    tray_reached: bool
    lowered_to_tray: bool
    released_over_tray: bool
    placement_success: bool


class TaskMetricsTracker:
    def __init__(
        self,
        *,
        controller: PickPlaceController,
        initial_cube_z: float,
    ) -> None:
        self.controller = controller
        self.initial_cube_z = initial_cube_z
        self.previous_cube_pos = controller.data.body("cube").xpos.copy()
        self.grasp_steps = 0
        self.lowering_steps = 0
        self.release_steps = 0
        self.placement_steps = 0
        self.retreat_steps = 0
        self.post_release_recontact = False
        self.grasped = False
        self.lifted = False
        self.tray_reached = False
        self.lowered_to_tray = False
        self.released_over_tray = False
        self.placement_success = False

    def update(self) -> None:
        cube_pos = self.controller.data.body("cube").xpos.copy()
        tray_pos = self.controller.data.site_xpos[self.controller.tray_center_id]
        tray_rot = self.controller.data.site_xmat[
            self.controller.tray_center_id
        ].reshape(3, 3)
        cube_tray_offset = tray_rot.T @ (cube_pos - tray_pos)
        grasp_pos = self.controller.data.site_xpos[self.controller.grasp_id]
        grasp_tray_offset = tray_rot.T @ (grasp_pos - tray_pos)
        two_finger_contact = self.controller.cube_has_two_finger_contact()
        any_finger_contact = self.controller.cube_has_any_finger_contact()

        self.grasp_steps = self.grasp_steps + 1 if two_finger_contact else 0
        if self.grasp_steps >= GRASP_STABLE_STEPS:
            self.grasped = True

        cube_lift = float(cube_pos[2] - self.initial_cube_z)
        if self.grasped and two_finger_contact and cube_lift >= CUBE_LIFT_MIN_DELTA:
            self.lifted = True

        tray_error = float(np.linalg.norm(cube_tray_offset[:2]))
        if self.lifted and tray_error <= TRAY_PLACE_TOL:
            self.tray_reached = True

        cube_inside_tray = bool(
            np.all(np.abs(cube_tray_offset[:2]) <= TRAY_INNER_XY_TOL)
        )
        cube_at_placement_height = (
            abs(float(cube_tray_offset[2])) <= PLACEMENT_Z_TOL
        )
        lowering_is_stable = (
            self.tray_reached
            and two_finger_contact
            and cube_inside_tray
            and cube_at_placement_height
        )
        self.lowering_steps = self.lowering_steps + 1 if lowering_is_stable else 0
        if self.lowering_steps >= LOWERING_STABLE_STEPS:
            self.lowered_to_tray = True

        release_is_stable = (
            self.lowered_to_tray
            and cube_inside_tray
            and cube_at_placement_height
            and not any_finger_contact
        )
        self.release_steps = self.release_steps + 1 if release_is_stable else 0
        if self.release_steps >= RELEASE_STABLE_STEPS:
            self.released_over_tray = True

        if (
            self.released_over_tray
            and not self.placement_success
            and any_finger_contact
        ):
            self.post_release_recontact = True

        cube_speed = float(
            np.linalg.norm(cube_pos - self.previous_cube_pos) / self.controller.dt
        )
        self.previous_cube_pos = cube_pos
        placement_is_stable = (
            self.released_over_tray
            and cube_inside_tray
            and cube_at_placement_height
            and not any_finger_contact
            and cube_speed <= PLACEMENT_MAX_SPEED
        )
        self.placement_steps = self.placement_steps + 1 if placement_is_stable else 0
        retreat_is_stable = (
            self.placement_steps >= PLACEMENT_STABLE_STEPS
            and not self.post_release_recontact
            and float(grasp_tray_offset[2]) >= RETREAT_MIN_HEIGHT
        )
        self.retreat_steps = self.retreat_steps + 1 if retreat_is_stable else 0
        if self.retreat_steps >= RETREAT_STABLE_STEPS:
            self.placement_success = True

    def result(self) -> TaskMetrics:
        return {
            "grasped": self.grasped,
            "lifted": self.lifted,
            "tray_reached": self.tray_reached,
            "lowered_to_tray": self.lowered_to_tray,
            "released_over_tray": self.released_over_tray,
            "placement_success": self.placement_success,
        }
