# WHAT? 
List of issues and tasks to be done.

## KEYS
 * -> IGNORED 
 + -> DONE
 - -> PENDING

## ISSUES 

* make site poses calculation consistent
* arm should stop moving before gripper closes

- move `control_loop` in `run_pick_place.py` to `PickPlaceController`
- add rollout.py
- fix collect_demos.py

- remove success checks from `run_pick_place.py`
+ refactor `sim.py` -> fix cube reset duplication logic


**`expert.py`**
- add `step` method in PickPlaceController

**`data.py`**
- test with/without velocity observations
- test `build_observation` implementation 
- test `build_action` implementation 
- normalization? 
- integrate episode success check
- add replay mechanism / logging / bag mechanism
- would slowing down walltime/sim-time effect the num frames in the dataset? 

** `infer.py` **
- actuator range clamping
- give time for sim to settle down