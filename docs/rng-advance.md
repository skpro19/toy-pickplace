# RNG Advance During Environment Initialization

This note explains why `SimEnv.__init__` should avoid consuming randomized scene samples.

## The Core Idea

An RNG is a sequence generator. When we create one with a seed, it produces a deterministic sequence of values.

For example, with `seed=0`, the RNG has a fixed sequence:

```text
sample 1, sample 2, sample 3, ...
```

Every time code asks the RNG for a random value, the RNG moves forward. This is called advancing or consuming the RNG state.

## Why This Matters For SimEnv

For randomized rollouts, we want this behavior:

```python
sim = SimEnv(randomize_scene=True, seed=0)
sim.reset_episode()
```

Expected behavior:

```text
episode 1 uses sample 1
```

But if `SimEnv.__init__` internally calls a randomized reset, then construction already consumes the first random scene.

That would make the sequence behave like this:

```text
SimEnv(...) construction uses sample 1
episode 1 uses sample 2
```

This is surprising because simply creating the environment changed which randomized layout episode 1 receives.

## Concrete Example

Imagine `SimEnv.__init__` did this:

```python
class SimEnv:
    def __init__(self, *, randomize_scene: bool, seed: int):
        self.randomize_scene = randomize_scene
        self.rng = np.random.default_rng(seed)
        self.reset_episode()
```

If `randomize_scene=True`, then `self.reset_episode()` samples a cube and tray position immediately.

So this code:

```python
sim = SimEnv(randomize_scene=True, seed=0)
sim.reset_episode()
```

does two randomized resets:

```text
1. hidden reset inside __init__
2. explicit reset requested by the caller
```

The caller probably thinks the first real episode starts at the first sampled layout, but it actually starts at the second sampled layout.

## Why This Is Bad

Hidden RNG advancement makes experiments harder to reason about.

Problems:

- `seed=0` no longer clearly means episode 1 gets the first sampled layout.
- Adding or removing initialization code can silently change all later randomized episodes.
- Reproducing a specific episode becomes harder.
- Data collection and rollout evaluation may disagree even if they use the same seed.

## Preferred Behavior

Environment construction should initialize MuJoCo state, but it should not consume a randomized episode layout.

Preferred sequence:

```python
sim = SimEnv(randomize_scene=True, seed=0)
```

This should do deterministic setup only.

Then:

```python
sim.reset_episode()
```

This should start episode 1 and consume the first randomized scene.

Expected sequence:

```text
SimEnv(...) construction consumes no scene sample
episode 1 uses sample 1
episode 2 uses sample 2
episode 3 uses sample 3
```

## How To Avoid Hidden RNG Advance

If `reset_episode()` always follows `self.randomize_scene`, then `__init__` should not call it directly when `randomize_scene=True`.

Instead, use a private deterministic initialization path.

Conceptually:

```python
class SimEnv:
    def __init__(self, *, randomize_scene: bool, seed: int):
        self.randomize_scene = randomize_scene
        self.rng = np.random.default_rng(seed)
        self._reset_deterministic()

    def reset_episode(self) -> None:
        self._reset_deterministic()
        if self.randomize_scene:
            self._apply_randomized_scene()
```

The important part is:

```text
__init__ initializes the simulator deterministically
reset_episode starts an actual episode and may randomize
```

## Summary

The issue is not that randomization during construction is physically wrong. The issue is that it consumes a random sample before the user-requested first episode.

For reproducible experiments, construction should not secretly advance the RNG. Episode resets should be the place where randomized scene samples are consumed.
