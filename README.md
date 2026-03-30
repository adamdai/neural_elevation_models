# Neural Elevation Models

Fresh restart of the NEMo codebase around a smaller core abstraction:

- `Nemo` is the user-facing entry point for fitting and querying a neural height field `z = h(x, y)`.
- Implementations are pluggable and share a common interface for height and gradient evaluation.
- Fitting is configurable and separated from the model definition.
- Large DEMs can be handled with overlapping local tiles.

## Current Layout

```text
nemo/
  baselines.py        # Simple base surfaces used by residual models
  fit.py              # Generic torch fitter and fit configuration
  height_field.py     # Abstract height-field interface
  nemo.py             # User-facing Nemo wrapper/factory
  tiling.py           # Overlapping local tile composition
  models/
    residual_mlp.py   # Baseline 1: residual MLP over normalized coordinates
    hashgrid.py       # tiny-cuda-nn hash-grid implementation
tests/
```

## Example

```python
import torch
from nemo import Nemo, PlaneBaseline, TorchFitConfig

xy = torch.rand(4096, 2) * 2.0 - 1.0
z = (
    0.2 * xy[:, :1]
    - 0.1 * xy[:, 1:2]
    + 0.05 * torch.sin(3.0 * xy[:, :1]) * torch.cos(2.0 * xy[:, 1:2])
)

nemo = Nemo.residual_mlp(
    bounds=((-1.0, 1.0), (-1.0, 1.0)),
    baseline=PlaneBaseline(),
    hidden_dim=128,
    depth=4,
)

nemo.fit(xy, z, fit_config=TorchFitConfig(iterations=800, lr=1e-3))
height = nemo.h(torch.tensor([[0.1, -0.3]]))
gradient = nemo.grad(torch.tensor([[0.1, -0.3]]))
```

## Notes

- `ResidualMLPHeightField` normalizes coordinates into `[-1, 1]^2` and learns a residual on top of a configurable baseline.
- `TCNNHashGridHeightField` is optional and requires `tinycudann`.
- `TiledHeightField` fits overlapping local models and blends them smoothly at inference time.
