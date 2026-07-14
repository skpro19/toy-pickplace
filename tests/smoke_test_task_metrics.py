from pathlib import Path
import sys

import numpy as np


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from rollout import (  # noqa: E402
    LOWERING_STABLE_STEPS,
    PLACEMENT_STABLE_STEPS,
    RELEASE_STABLE_STEPS,
    TaskMetricsTracker,
)


class FakeBody:
    def __init__(self, *, xpos: np.ndarray) -> None:
        self.xpos = xpos


class FakeData:
    def __init__(self) -> None:
        self.cube_pos = np.zeros(3, dtype=np.float64)
        self.site_xpos = np.array([[0.2, 0.1, 0.05]], dtype=np.float64)
        angle = np.pi / 2.0
        tray_rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.site_xmat = tray_rot.reshape(1, 9)

    def body(self, name: str) -> FakeBody:
        if name != "cube":
            raise ValueError(f"Unexpected body: {name}")
        return FakeBody(xpos=self.cube_pos)


class FakeController:
    def __init__(self) -> None:
        self.data = FakeData()
        self.tray_center_id = 0
        self.dt = 1.0
        self.two_finger_contact = False
        self.any_finger_contact = False

    def cube_has_two_finger_contact(self) -> bool:
        return self.two_finger_contact

    def cube_has_any_finger_contact(self) -> bool:
        return self.any_finger_contact


def set_cube_tray_offset(
    *,
    controller: FakeController,
    local_offset: np.ndarray,
) -> None:
    tray_pos = controller.data.site_xpos[controller.tray_center_id]
    tray_rot = controller.data.site_xmat[controller.tray_center_id].reshape(3, 3)
    controller.data.cube_pos = tray_pos + tray_rot @ local_offset


def main() -> None:
    no_lowering_controller = FakeController()
    no_lowering_tracker = TaskMetricsTracker(
        controller=no_lowering_controller,
        initial_cube_z=0.0,
    )
    no_lowering_tracker.lifted = True
    no_lowering_tracker.tray_reached = True
    set_cube_tray_offset(
        controller=no_lowering_controller,
        local_offset=np.array([0.04, 0.0, 0.03], dtype=np.float64),
    )
    for _ in range(PLACEMENT_STABLE_STEPS):
        no_lowering_tracker.update()
    assert not no_lowering_tracker.result()["placement_success"]

    controller = FakeController()
    tracker = TaskMetricsTracker(controller=controller, initial_cube_z=0.0)

    controller.two_finger_contact = True
    controller.any_finger_contact = True
    for _ in range(5):
        tracker.update()
    assert tracker.result()["grasped"]

    controller.data.cube_pos = np.array([0.0, 0.0, 0.05], dtype=np.float64)
    tracker.update()
    assert tracker.result()["lifted"]

    set_cube_tray_offset(
        controller=controller,
        local_offset=np.array([0.051, 0.0, 0.03], dtype=np.float64),
    )
    tracker.update()
    assert tracker.result()["tray_reached"]
    assert not tracker.result()["lowered_to_tray"]

    inside_offset = np.array([0.04, 0.0, 0.03], dtype=np.float64)
    for _ in range(LOWERING_STABLE_STEPS - 1):
        set_cube_tray_offset(controller=controller, local_offset=inside_offset)
        tracker.update()
    assert not tracker.result()["lowered_to_tray"]

    set_cube_tray_offset(
        controller=controller,
        local_offset=np.array([0.051, 0.0, 0.03], dtype=np.float64),
    )
    tracker.update()
    for _ in range(LOWERING_STABLE_STEPS):
        set_cube_tray_offset(controller=controller, local_offset=inside_offset)
        tracker.update()
    assert tracker.result()["lowered_to_tray"]

    controller.two_finger_contact = False
    controller.any_finger_contact = False
    set_cube_tray_offset(
        controller=controller,
        local_offset=np.array([0.051, 0.0, 0.03], dtype=np.float64),
    )
    for _ in range(RELEASE_STABLE_STEPS):
        tracker.update()
    assert not tracker.result()["released_over_tray"]

    for _ in range(RELEASE_STABLE_STEPS - 1):
        set_cube_tray_offset(controller=controller, local_offset=inside_offset)
        tracker.update()
    assert not tracker.result()["released_over_tray"]

    controller.any_finger_contact = True
    tracker.update()
    controller.any_finger_contact = False
    for _ in range(RELEASE_STABLE_STEPS):
        tracker.update()
    assert tracker.result()["released_over_tray"]
    assert not tracker.result()["placement_success"]

    for _ in range(PLACEMENT_STABLE_STEPS - 1):
        tracker.update()
    assert tracker.result()["placement_success"]

    controller.data.cube_pos = np.zeros(3, dtype=np.float64)
    tracker.update()
    assert tracker.result()["lowered_to_tray"]
    assert tracker.result()["released_over_tray"]
    assert tracker.result()["placement_success"]

    print("Task metrics smoke test passed.")


if __name__ == "__main__":
    main()
