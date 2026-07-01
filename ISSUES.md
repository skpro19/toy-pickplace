# What? 
List of issues and tasks to be done.

## Keys 
 * -> IGNORED 
 + -> DONE
 - -> PENDING

## Issues

* make site poses calculation consistent
* arm should stop moving before gripper closes

- consolidate the `control() → mj_step() → update_phase()` loop into a `step()` method on `PickPlaceController` to avoid duplication between `run_viewer()` and `run_headless()`
- refactor `pick_place_controller.py` to carve out the expert
- add rollout.py
- fix collect_demos.py