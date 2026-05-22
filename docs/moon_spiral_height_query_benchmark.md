# Moon Spiral Height Query Benchmark

Date: May 21, 2026

Dataset: `/home/shared/data_nerfstudio/moon_spiral_2_masked/dem_pc.npy`

NEMo checkpoint: `/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000/nemo_model.pt`

Hardware:

- CPU: Intel Core i7-14700K
- GPU: NVIDIA GeForce RTX 4090, 24 GiB
- Environment: `conda run -n nemo`

The GT DEM is a regular `2000 x 2000` raster with approximately `1.0005 m` spacing. Timings below are height-only queries at random in-bounds `(x, y)` points. Values are average milliseconds per batch; lower is faster.

## Selected Results

| Method | N=1 | N=1k | N=10k | N=100k | N=1M |
|---|---:|---:|---:|---:|---:|
| NEMo height CPU | 0.0596 ms | 0.720 ms | 1.061 ms | 14.83 ms | 201.4 ms |
| NEMo height CUDA | 0.650 ms | 0.595 ms | 0.528 ms | 1.248 ms | 10.68 ms |
| Repo `DEM.query` bilinear | 0.0274 ms | 0.175 ms | 1.185 ms | 14.56 ms | 165.8 ms |
| SciPy `RegularGridInterpolator` linear | 0.0191 ms | 0.114 ms | 1.083 ms | 11.57 ms | 116.3 ms |
| SciPy `ndimage.map_coordinates` order 1 | 0.0091 ms | 0.0599 ms | 0.224 ms | 3.571 ms | 43.19 ms |
| Torch `grid_sample` bilinear CPU | 0.0268 ms | 0.0382 ms | 0.0882 ms | 0.569 ms | 6.021 ms |
| Torch `grid_sample` bilinear CUDA | 0.372 ms | 0.460 ms | 0.482 ms | 0.221 ms | 0.915 ms |
| Torch `grid_sample` bicubic CUDA | 0.485 ms | 0.474 ms | 0.483 ms | 0.213 ms | 1.047 ms |

## Full Method Notes

- `DEM.query` is the repo's manual bilinear implementation. It is clean and general, but slower than methods that exploit uniform raster coordinates directly.
- `RegularGridInterpolator` is convenient and general for rectilinear grids, but is not the fastest option for this uniform DEM.
- `ndimage.map_coordinates` is a good CPU choice for uniform-grid interpolation. Order 0 is nearest, order 1 is bilinear, and order 3 is cubic spline interpolation.
- Cubic `ndimage.map_coordinates` should prefilter once and then query with `prefilter=False`. Recomputing the spline filter every query is much slower.
- Torch `grid_sample` is the fastest path in these tests for large batches, especially on CUDA. It also supports bilinear and bicubic interpolation with one API.
- NEMo CUDA is much faster than NEMo CPU for large batches, but still slower than querying the DEM raster with `grid_sample` because NEMo runs an MLP plus residual grid rather than a direct raster lookup.

## Cubic Interpolation Detail

SciPy cubic interpolation has two very different timings depending on whether the spline prefilter is recomputed every call.

| Method | N=1 | N=1k | N=10k | N=100k | N=1M |
|---|---:|---:|---:|---:|---:|
| `ndimage` order 3, prefilter every call | 50.9 ms | 52.1 ms | 50.9 ms | 62.3 ms | 178.7 ms |
| `ndimage` order 3, prefiltered once | 0.0033 ms | 0.0461 ms | 0.508 ms | 5.60 ms | 60.5 ms |
| Torch `grid_sample` bicubic CPU | 0.0258 ms | 0.0382 ms | 0.189 ms | 1.459 ms | 13.91 ms |
| Torch `grid_sample` bicubic CUDA | 0.485 ms | 0.474 ms | 0.483 ms | 0.213 ms | 1.047 ms |

The practical takeaway is to avoid per-query cubic prefiltering. For repeated cubic DEM queries, precompute spline coefficients or use `grid_sample`.

## Throughput at Large Batch Sizes

For `N=1,000,000` height queries:

| Method | Time | Throughput |
|---|---:|---:|
| DEM torch `grid_sample` bilinear CUDA | 0.915 ms | 1.09 billion queries/s |
| DEM torch `grid_sample` bicubic CUDA | 1.047 ms | 955 million queries/s |
| DEM torch `grid_sample` bilinear CPU | 6.021 ms | 166 million queries/s |
| NEMo height CUDA | 10.68 ms | 93.6 million queries/s |
| SciPy `ndimage` order 1 | 43.19 ms | 23.2 million queries/s |
| SciPy `RegularGridInterpolator` linear | 116.3 ms | 8.6 million queries/s |
| Repo `DEM.query` bilinear | 165.8 ms | 6.0 million queries/s |
| NEMo height CPU | 201.4 ms | 5.0 million queries/s |

## Interpretation

For one-off scalar queries, CPU interpolation has the lowest latency. CUDA has launch and transfer overhead, so it is not worthwhile for `N=1` or very small batches.

For path planning batches around `1k` to `10k` samples, CPU `grid_sample`, SciPy `ndimage` order 1, or direct NumPy bilinear are all fast. NEMo CPU is acceptable but slower; NEMo CUDA becomes competitive once the batch is at least a few thousand points.

For dense map queries or rendering-scale workloads, the DEM raster is much faster than NEMo. The fastest measured route is torch `grid_sample` on CUDA. If the path planner already runs on torch tensors, representing the DEM as a torch raster and querying it with `grid_sample` is the cleanest high-throughput option.

The comparison is speed-only. NEMo and GT DEM queries are not equivalent semantically: NEMo is a learned compact approximation, while DEM interpolation queries the reference terrain directly.
