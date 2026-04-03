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
  dem.py              # Standardized DEM grid interface
  fit.py              # Generic torch fitter and fit configuration
  height_field.py     # Abstract height-field interface
  io/
    dem_loaders.py    # Format-specific DEM readers
  nemo.py             # User-facing Nemo wrapper/factory
  tiling.py           # Overlapping local tile composition
  models/
    residual_mlp.py   # Baseline 1: residual MLP over normalized coordinates
    hashgrid.py       # tiny-cuda-nn hash-grid implementation
tests/
scripts/
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

- `DEM` standardizes loaded terrain grids and exposes `query`, `grad`, `crop`, and index/coordinate conversions.
- DEM loading supports `.npy`, ASCII grid `.asc`, GeoTIFF `.tif`/`.tiff`, and `.dat` files containing either a pickled NumPy grid array or text data in regular `x y z` triplets or a whitespace-delimited `z` matrix.
- `ResidualMLPHeightField` normalizes coordinates into `[-1, 1]^2` and learns a residual on top of a configurable baseline.
- `SmoothGridHeightField` uses a configurable smooth backbone (`mlp` or `siren`) plus an optional residual term (`grid` or `none`).
- `TCNNHashGridHeightField` is optional and requires `tinycudann`.
- `TiledHeightField` fits overlapping local models and blends them smoothly at inference time.

## CLI

Fit a NEMo directly to a DEM:

```bash
python scripts/fit_dem.py --dem-path path/to/dem.npy --iterations 1500 --lr 1e-3
```

This writes outputs by default to `output/<dem_name>_<model>/`, including `fit.html` and `results.json`.

`fit_dem.py` also supports loading spatial DEM patches through either explicit `--xlims ... --ylims ...` bounds or a named `--patch-name` preset from `data/dem_patches.json`. For `Mt_Etna-DSM.tif`, the default behavior is to load the `s3li_crater_dem_buffer_5` patch when no crop is provided.

Select a model family with tyro subcommands:

```bash
python scripts/fit_dem.py --dem-path path/to/dem.npy model:residual-mlp --hidden-dim 256 --depth 5
python scripts/fit_dem.py --dem-path path/to/dem.npy model:tiled-residual-mlp --tile-size-x 128 --tile-size-y 128
python scripts/fit_dem.py --dem-path path/to/dem.npy model:smooth-grid --model.hidden-dim 64 --model.grid-resolution-x 128 --model.grid-resolution-y 128
python scripts/fit_dem.py --dem-path path/to/dem.npy model:smooth-grid --model.backbone-type siren --model.residual-type grid --model.siren-omega-0 30.0
python scripts/fit_dem.py --dem-path path/to/dem.npy model:smooth-grid --model.color-type hashgrid
python scripts/fit_dem.py --dem-path path/to/dem.npy model:hashgrid
```

Interactive viewer for saved checkpoints:

```bash
pip install -e ".[viewer]"
python scripts/mesh_viewer.py --checkpoint-path path/to/model.pt
```

This viewer samples the NeMO into a colored terrain mesh and lets the browser rasterize it quickly.

Exact ray-based viewer for training/debugging:

```bash
python scripts/training_viewer.py --checkpoint-path path/to/model.pt
```

The exact viewer defaults to a lower-cost draft render preset tuned for responsiveness. Increase `--num-bracket-samples`, `--num-bisection-steps`, or `--num-newton-steps` if you want a higher-quality render.

Run a lightweight architecture and hyperparameter search over `fit_dem.py`:

```bash
python scripts/autotune_dem.py --dem-path path/to/dem.npy --trials 12
```

This keeps each trial in its own output folder under `output/autotune/<study_name>/trials/`, reuses `fit_dem.py` for training/evaluation, and writes a study summary plus leaderboard for comparing runs. By default the score is a weighted combination of height RMSE, gradient RMSE, height max error, and gradient max error.
