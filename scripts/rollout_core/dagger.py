"""Pure DAgger intervention selection."""

import numpy as np


DAGGER_INTERVENTION_MODES = ("beta", "threshold")
DEFAULT_INTERVENTION_STEPS = 50


def select_dagger_control(
    *,
    mode: str,
    beta: float,
    arm_disagreement: float,
    gripper_disagreement: bool,
    intervention_threshold: float | None,
    intervention_steps: int,
    intervention_steps_remaining: int,
    rng: np.random.Generator,
) -> tuple[bool, int]:
    if mode == "beta":
        return rng.random() < beta, 0
    if mode != "threshold":
        raise ValueError(f"Unsupported DAgger intervention mode: {mode!r}")
    if intervention_threshold is None:
        raise ValueError("Threshold intervention mode requires a threshold")

    intervention_triggered = (
        arm_disagreement > intervention_threshold or gripper_disagreement
    )
    if intervention_triggered and intervention_steps_remaining == 0:
        intervention_steps_remaining = intervention_steps

    execute_expert = intervention_steps_remaining > 0
    if execute_expert:
        intervention_steps_remaining -= 1
    return execute_expert, intervention_steps_remaining
