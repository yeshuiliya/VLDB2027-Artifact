# Input dataset conventions

Multi-view matrices must represent the same sample IDs in the same order.

- `X1, X2, ...` with `Y` supports `.npz` and `.mat` (classic and v7.3).
- For MATLAB `.mat`, a cell array `X` plus a label vector `Y` is also supported.
- `Y` may use 0-based or 1-based integer labels; labels are reindexed internally.
- Features must be finite, numeric, and two-dimensional.
- Transposed view matrices `(D, N)` are inferred from the label count.
- Default preprocessing scales each raw view independently, feature-wise to `[0, 1]`.
- This pipeline assumes aligned samples across all available views.

Example:

```python
import numpy as np
from scipy.io import savemat

N = 100
rng = np.random.default_rng(0)
savemat('dataset.mat', {
  'X1': rng.normal(size=(N, 16)),
  'X2': rng.normal(size=(N, 32)),
  'Y': rng.integers(0, 5, size=(N, 1)),
})
```

**Dataset permissions:** use only data that you are allowed to access and redistribute. The example generator creates synthetic data locally.
