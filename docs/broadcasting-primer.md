# PyTorch reductions and broadcasting primer

Notes on how tensor **collapsing** works during reductions (`mean`, `std`, `sum`, …), how to control it with `dim` and `keepdim`, and how that connects to **broadcasting** — especially when normalizing action batches in training.

---

## Context: action normalization in training

In `scripts/train.py`, a batch of actions has shape `(batch_size, 8)` — 7 arm joint controls plus 1 gripper control. When computing normalization stats over the arm slice:

```python
actions[:, 0:7]   # shape (batch_size, 7)
```

The choice of `dim` (or omitting it) determines whether you get one scalar, one stat per joint, or one stat per sample.

---

## Part 1: One scalar vs per-joint stats

### Setup: a tiny batch

Imagine **3 samples** and **2 arm joints** (instead of 32 samples and 7 joints — same idea, easier math):

```text
actions[:, 0:2] =

        joint0   joint1
sample0   10       100
sample1   20       200
sample2   30       300
```

Shape is `(3, 2)` → 3 rows (batch), 2 columns (joints).

Joint0 values are small (10, 20, 30). Joint1 values are large (100, 200, 300). Different joints can live on very different scales.

### Option A: `torch.mean(...)` with no `dim` → one scalar

PyTorch flattens everything into one list:

```text
[10, 100, 20, 200, 30, 300]
```

Then averages all 6 numbers:

```text
(10 + 100 + 20 + 200 + 30 + 300) / 6
= 660 / 6
= 110
```

**Result: one number — `110`**

That single `110` would be used for **both** joint0 and joint1 if you normalize with it. But joint0 is really around 10–30, and joint1 is around 100–300. One shared mean does not describe either joint well.

This is what **collapsing** means: batch and joints are merged into one statistic.

### Option B: `.mean(dim=0)` → one mean per joint

`dim=0` means: for each column, average down the rows.

**Joint 0 column:** 10, 20, 30  
→ mean = `(10 + 20 + 30) / 3 = 20`

**Joint 1 column:** 100, 200, 300  
→ mean = `(100 + 200 + 300) / 3 = 200`

**Result: two numbers — `[20, 200]`**

Each joint gets its own mean.

### Side-by-side picture

```text
        joint0   joint1
s0        10      100
s1        20      200
s2        30      300

mean (no dim):     110          ← one number for everything

mean (dim=0):      20    200    ← one number per joint
                   ↑      ↑
                 joint0 joint1
```

### What normalization would look like

Say you subtract the mean (ignoring std for now).

**With scalar mean (110):**

```text
sample0, joint0:  10 - 110 = -100   ← was 10, now huge negative
sample0, joint1: 100 - 110 =  -10
```

Joint0 gets crushed because 110 is dominated by joint1's large values.

**With per-joint mean `[20, 200]`:**

```text
sample0, joint0:  10 -  20 = -10   ← reasonable
sample0, joint1: 100 - 200 = -100  ← reasonable for that joint
```

Each dimension is centered around **its own** typical value.

### Code example (tiny tensor)

```python
import torch

actions = torch.tensor([
    [10., 100.],   # sample 0: arm joints 0 and 1
    [20., 200.],   # sample 1
    [30., 300.],   # sample 2
])  # shape (3, 2) — like actions[:, 0:2]

# Collapse everything → one scalar
scalar_mean = actions.mean()
print(scalar_mean)           # tensor(110.)

# Keep one stat per joint → shape (2,)
per_joint_mean = actions.mean(dim=0)
print(per_joint_mean)        # tensor([ 20., 200.])
```

### Mapping to real training shapes

After `actions[:, 0:7]`, the tensor is `(32, 7)` with `batch_size=32`:

| What you write | What gets averaged | Result |
|----------------|-------------------|--------|
| `actions[:, 0:7].mean()` | all 32×7 = **224** values | **1** scalar |
| `actions[:, 0:7].mean(dim=0)` | 32 values **per column** | **7** numbers (one per joint) |

**One-sentence summary:** Collapsing happens because `mean()` with no `dim` treats the whole `(batch, joints)` block as one pile of numbers. `mean(dim=0)` only averages across the batch and keeps a separate statistic for each joint.

---

## Part 2: How collapsing works in PyTorch

In PyTorch, **collapsing** means a dimension **disappears** after a reduction. The tensor gets **smaller rank** (fewer dimensions).

### The rule

```python
x.shape  # before
y = x.mean(dim=?)  # that dimension is removed unless you opt out
y.shape  # after — one fewer dim (usually)
```

| Call | Effect |
|------|--------|
| `x.mean()` | Collapse **all** dims → scalar |
| `x.mean(dim=0)` | Collapse **dim 0** only |
| `x.mean(dim=0, keepdim=True)` | Collapse dim 0 **but keep it as size 1** |

### Tiny example

```python
import torch

x = torch.tensor([
    [10., 100.],
    [20., 200.],
    [30., 300.],
])
# shape: (3, 2)  →  dim 0 = batch, dim 1 = joints
```

**Collapse everything:**

```python
x.mean()           # tensor(110.)     shape: ()
```

All 6 values → one number. Both dimensions gone.

**Collapse one dimension:**

```python
x.mean(dim=0)      # tensor([20., 200.])   shape: (2,)
# averages down rows → one stat per joint

x.mean(dim=1)      # tensor([ 55., 110., 175.])   shape: (3,)
# averages across joints → one stat per sample
```

Each call removes **one** dimension.

**Prevent shape surprises: `keepdim=True`**

```python
x.mean(dim=0, keepdim=True)   # shape: (1, 2)  not (2,)
x.mean(dim=1, keepdim=True)   # shape: (3, 1)  not (3,)
```

The reduced dimension stays as **1**, which helps broadcasting (see Part 3).

### Mental model: `dim` = "which axis to squeeze out"

```text
shape (batch, joints)

dim=0  →  collapse batch   →  shape (joints,)
dim=1  →  collapse joints  →  shape (batch,)
no dim →  collapse all     →  shape ()
```

Think: **"average along this axis"** — that axis is removed.

### How to prevent unwanted collapsing

| Goal | Do this |
|------|---------|
| Stats **per feature/joint** | `.mean(dim=0)` on `(batch, features)` |
| Stats **per sample** | `.mean(dim=1)` |
| Keep ranks for broadcasting | add `keepdim=True` |
| Never collapse to scalar by accident | **always pass `dim=`** when you mean axis-wise stats |
| Keep a length-1 axis explicitly | `keepdim=True` or `.unsqueeze(dim)` after |

**Practical pattern for action normalization:**

```python
# Per-joint mean/std over the batch — what you usually want
arm_mean = actions[:, 0:7].mean(dim=0)           # (7,)
arm_std  = actions[:, 0:7].std(dim=0, unbiased=False)

# Normalize (broadcasts automatically)
actions[:, 0:7] = (actions[:, 0:7] - arm_mean) / arm_std
```

If you want `(1, 7)` instead of `(7,)`:

```python
arm_mean = actions[:, 0:7].mean(dim=0, keepdim=True)  # (1, 7)
```

### Quick cheat sheet

```python
t.shape          # (B, F)  batch × features

t.mean()         # ()           scalar — collapsed ALL
t.mean(dim=0)    # (F,)         one value per feature
t.mean(dim=1)    # (B,)         one value per sample
t.mean(dim=0, keepdim=True)  # (1, F)
t.mean(dim=1, keepdim=True)  # (B, 1)
```

**One-liner:** Collapsing = a dimension goes away after reduction. **Pass `dim=`** to control which axis disappears; **`keepdim=True`** to keep it as size 1 instead of removing it entirely.

---

## Part 3: Why `keepdim=True` helps broadcasting

Broadcasting compares shapes **from the right** (last dimension first). A reduced tensor has **fewer dimensions**, so PyTorch has to infer which axis each remaining size belongs to. Sometimes that inference is wrong.

### Case 1: reduce `dim=0` (batch) — often works without `keepdim`

```python
x.shape           # (32, 7)  batch × joints
m = x.mean(dim=0) # (7,)     one mean per joint

(x - m).shape     # (32, 7)  ✓ works
```

Why: `(7,)` lines up with the **last** dim of `(32, 7)`:

```text
(32, 7)
   (7,)   ← aligned with joints
```

For per-joint stats over the batch, `keepdim=True` is optional — mainly for clarity.

### Case 2: reduce `dim=1` (features) — `keepdim` is often required

```python
x.shape           # (32, 7)
m = x.mean(dim=1) # (32,)    one mean per sample

(x - m)           # ✗ error!
```

PyTorch tries to align `(32,)` with the **last** dim:

```text
(32, 7)
 (32,)   ← PyTorch thinks: 7 vs 32 → mismatch
```

You wanted "subtract one number **per row**," but `(32,)` looks like it belongs on the **joint** axis, not the batch axis.

**Fix with `keepdim=True`:**

```python
m = x.mean(dim=1, keepdim=True)  # (32, 1)

(x - m).shape   # (32, 7)  ✓
```

```text
(32, 7)
(32, 1)  ← clearly "one value per row, broadcast across columns"
```

### What `keepdim=True` actually does

It does not change the math — same averages — but keeps the reduced axis as size **1** instead of removing it:

```python
# dim=0
x.mean(dim=0)                # (7,)
x.mean(dim=0, keepdim=True)  # (1, 7)

# dim=1
x.mean(dim=1)                # (32,)
x.mean(dim=1, keepdim=True)  # (32, 1)
```

The result still "points" along the axis you reduced, which makes broadcasting unambiguous.

### Visual: per-row mean

```text
x =
     j0   j1   j2
s0 [ 10,  20,  30 ]  → row mean = 20
s1 [ 40,  50,  60 ]  → row mean = 50

You want:
s0: [10-20, 20-20, 30-20] = [-10, 0, 10]
s1: [40-50, 50-50, 60-50] = [-10, 0, 10]
```

Need shape `(2, 1)` subtracted from `(2, 3)`:

```python
row_mean = x.mean(dim=1, keepdim=True)  # [[20], [50]]
x - row_mean                            # works
```

`(2,)` would **not** broadcast correctly against `(2, 3)`.

### Rule of thumb

| You reduced… | Want to use result on full `(B, F)` tensor |
|--------------|-----------------------------------------------|
| **dim=0** (batch → per-feature stats) | `(F,)` usually fine; `(1, F)` with `keepdim` is clearer |
| **dim=1** (features → per-sample stats) | use **`keepdim=True`** → `(B, 1)` |
| **Higher rank** (images, sequences) | **`keepdim=True`** almost always — ambiguity grows fast |

---

## Part 4: Why `(32, 7) - (32, 1)` works but `(32, 7) - (32,)` fails

The rule is not "rightmost dims must be **equal**." Each dimension pair is OK if:

1. they are **equal**, or
2. **one of them is 1**

If every pair passes, broadcast. Any dim of size **1** is **stretched** to match the other.

### Why `(32, 7) - (32, 1)` works

```text
(32,  7)
(32,  1)
     ↑
   7 vs 1  → one is 1  ✓  →  1 becomes 7

(32,  7)
(32,  1)
 ↑
32 vs 32  → equal  ✓
```

So `(32, 1)` is treated as "same 32 rows, **one** column that gets copied across all 7 columns."

Conceptually:

```text
[[20],     →  [[20, 20, 20, 20, 20, 20, 20],
 [50]]         [50, 50, 50, 50, 50, 50, 50]]
```

Same row mean subtracted from every joint in that row.

### Why `(32, 7) - (32,)` fails

```text
(32, 7)
   (32)    ← only ONE dimension; aligns with the RIGHTMOST dim only
     ↑
   7 vs 32  → neither is 1  ✗
```

`(32,)` is not "one value per row." PyTorch only sees a 1D tensor and lines it up with the **last** axis (joints), so it tries to match 7 with 32 — no broadcast via 1, so error.

To mean "per row," you need that information on the **batch** axis:

- `(32, 1)` with `keepdim=True`, or
- `(32, 1)` via `.unsqueeze(-1)` / `[:, None]`

### The "1 is special" idea

| Pair | Result |
|------|--------|
| `7` and `7` | ✓ equal |
| `7` and `1` | ✓ broadcast 1 → 7 |
| `1` and `7` | ✓ broadcast 1 → 7 |
| `7` and `32` | ✗ neither equal nor 1 |

So `(32, 1)` does not need `7 == 1`. It needs **`1` is allowed to become `7`**.

### Same rule for per-joint normalization

```text
(32, 7)
    (7)    ← 1D aligns with last dim
     ↑
   7 vs 7  ✓ equal
```

And for `(32, 7) - (1, 7)`:

```text
(32, 7)
( 1, 7)
 ↑
32 vs 1  ✓  →  1 becomes 32
```

### Short summary

- **Equal** → fine
- **One side is 1** → fine (stretch that axis)
- **Different and neither is 1** → error

`(32, 1)` works with `(32, 7)` because **`1` broadcasts to `7`**, not because `7` and `1` are equal. `(32,)` fails because it is aligned with the wrong axis and **`32 ≠ 7`** with no `1` to save it.

---

## Related notes for this project

- Actions are shape `(T, 8)`: indices `0:7` are arm controls, index `7` is gripper (see `scripts/sim.py`).
- For normalization, compute mean/std **once over the full dataset** (not per-batch) and save stats with the checkpoint for inference.
- `torch.std` defaults to unbiased std (`ddof=1`); use `unbiased=False` to match `numpy.std` default or for stable normalization on small batches.
- Gripper values (~0–255) and joint angles have very different scales — normalize arm and gripper separately if you enable action normalization.
