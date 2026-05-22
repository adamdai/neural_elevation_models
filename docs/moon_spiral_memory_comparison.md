# Moon Spiral Memory Comparison

Date: May 21, 2026

Dataset inspected: `/home/shared/data_nerfstudio/moon_spiral_2_masked`

Primary output inspected: `/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto/2026-05-16_threshold-vertical-fix_5000`

Note: `/home/shared/data_nerfstudio/moon_spiral` was not present on disk during this inspection, so the measurements below use `moon_spiral_2_masked`.

## Summary

The image data dominates loaded training memory. The main 100 PNG images occupy 385 MiB on disk, but expand to 791 MiB when decoded as uint8 RGBA, or 2.37 GiB if held as float32 RGB tensors.

The GT DEM point cloud is moderate in raw form: 4,000,000 XYZ float64 points, 91.6 MiB. If treated as the raster it actually is, the height-only representation is much smaller: 30.5 MiB as float64, 15.3 MiB as float32, or 7.6 MiB as uint16/int16 quantized heights before compression.

The learned NEMo height field is tiny compared with the radiance field. The standalone `nemo_model.pt` is about 103 KiB on disk and contains only 24,963 tensor parameters, about 0.095 MiB. A single Nerfstudio checkpoint is about 168 MiB on disk.

For a realistic mission-side terrain product, a 10 cm quantized height raster with origin, spacing, vertical offset, and scale metadata is a good default: it compressed to 1.26 MiB as tiled ZSTD GeoTIFF for this DEM, with 2.9 cm RMSE and less than 5 cm max height error relative to the float64 DEM heights.

## Dataset Footprint

| Item | Disk size | Loaded or raw memory | Notes |
|---|---:|---:|---|
| Full masked dataset folder | 1.6 GiB | variable | Includes images, COLMAP alignment copies, DEM point clouds |
| Main image set: `images/*.png` | 385 MiB | 791 MiB uint8 RGBA, 2.37 GiB float32 RGB | 100 images, each 1920 x 1080 |
| COLMAP full-res images | 460 MiB | 791 MiB uint8 RGBA, 2.37 GiB float32 RGB | Another 100 full-res images |
| COLMAP `images_2` | 119 MiB | 198 MiB uint8 RGBA, 593 MiB float32 RGB | 100 images, 960 x 540 |
| COLMAP `images_4` | 28.8 MiB | 49.4 MiB uint8 RGBA, 148 MiB float32 RGB | 100 images, 480 x 270 |
| COLMAP `images_8` | 7.0 MiB | 12.4 MiB uint8 RGBA, 37.0 MiB float32 RGB | 100 images, 240 x 135 |
| GT DEM point cloud: `dem_pc.npy` | 91.6 MiB | 91.6 MiB | Shape `(4_000_000, 3)`, float64 |
| COLMAP point cloud: `colmap_pc.npy` | 1.23 MiB | 1.23 MiB | Shape `(53_558, 3)`, float64 |

## Model and Output Footprint

Example checkpoint: `2026-05-16_threshold-vertical-fix_5000/nerfstudio_models/step-000019999.ckpt`

| Item | Disk size | Tensor memory | Notes |
|---|---:|---:|---|
| Nerfstudio checkpoint file | 168 MiB | 68.5 MiB model tensors + 105.8 MiB optimizer tensors | Includes training state |
| Main radiance field tensors | inside checkpoint | 46.6 MiB | `_model.field.*` |
| Proposal network tensors | inside checkpoint | 12.4 MiB | `_model.proposal_networks.*` |
| Other model tensors | inside checkpoint | 9.43 MiB | Misc model state |
| Camera optimizer tensors | inside checkpoint | 0.002 MiB | 588 parameters |
| Height field tensors inside checkpoint | inside checkpoint | 0.095 MiB | 24,963 parameters |
| Standalone NEMo height model: `nemo_model.pt` | 103 KiB | 0.095 MiB | Same height-field tensor payload |

Across `/home/shared/outputs/addai/moon_spiral_2_masked/terrain-nerfacto`, repeated experiment artifacts are the main storage cost:

| Artifact type | Count | Total size |
|---|---:|---:|
| `.ckpt` files | 35 | 5.34 GiB |
| `nemo_model.pt` files | 28 | 2.8 MiB |
| PNG render/plot artifacts | 4,460 | 1.73 GiB |
| HTML Plotly artifacts | 38 | 227 MiB |
| JSON files | 49 | 504 KiB |

## GT DEM Raster Structure

The GT DEM point cloud is already a regular raster grid:

| Property | Value |
|---|---:|
| Points | 4,000,000 |
| Grid shape | 2000 x 2000 |
| X range | -1000.0 m to 1000.0 m |
| Y range | -1000.0 m to 1000.0 m |
| Grid spacing | 1.000500250125 m |
| Z min | -42.171206 m |
| Z max | 83.035019 m |
| Z mean | 23.420581 m |
| Z standard deviation | 27.015585 m |

Because x and y are regular, storing all XYZ points is redundant. A raster representation only needs:

- a 2000 x 2000 height array,
- origin,
- x/y spacing,
- vertical offset and scale if quantized,
- coordinate frame metadata.

## GT DEM Raster Memory

| Raster representation | Raw memory | Loss relative to float64 DEM |
|---|---:|---:|
| Height-only float64 raster | 30.5 MiB | None |
| Height-only float32 raster | 15.3 MiB | RMSE 0.0000009 m, max 0.0000038 m |
| Height-only float16 raster | 7.6 MiB | RMSE 0.00735 m, max 0.0311 m |
| uint16, 1 cm quantization | 7.6 MiB | RMSE 0.00289 m, max 0.00498 m |
| uint16, 5 cm quantization | 7.6 MiB | RMSE 0.0144 m, max 0.0249 m |
| uint16, 10 cm quantization | 7.6 MiB | RMSE 0.0289 m, max 0.0498 m |
| uint16, 25 cm quantization | 7.6 MiB | RMSE 0.0721 m, max 0.1245 m |
| uint16, 50 cm quantization | 7.6 MiB | RMSE 0.1443 m, max 0.2490 m |
| uint16, 1 m quantization | 7.6 MiB | RMSE 0.2887 m, max 0.4981 m |

Quantized rasters use:

```text
stored_value = round((height_m - z_offset_m) / z_scale_m)
height_m = stored_value * z_scale_m + z_offset_m
z_offset_m = -42.17120622568093
```

## GT DEM Compression Results

These are measured on the actual 2000 x 2000 DEM height raster.

| Representation | Compressed format | Size | RMSE | Max abs error |
|---|---|---:|---:|---:|
| float64 height raster | `.npy` + `zstd -19` | 6.15 MiB | 0 m | 0 m |
| float32 height raster | `.npy` + `zstd -19` | 5.79 MiB | 0.0000009 m | 0.0000038 m |
| float32 height raster | tiled GeoTIFF ZSTD | 5.05 MiB | 0.0000009 m | 0.0000038 m |
| float16 height raster | `.npy` + `zstd -19` | 3.87 MiB | 0.00735 m | 0.0311 m |
| uint16, 1 cm | tiled GeoTIFF ZSTD | 3.19 MiB | 0.00289 m | 0.00498 m |
| uint16, 5 cm | tiled GeoTIFF ZSTD | 1.75 MiB | 0.0144 m | 0.0249 m |
| uint16, 10 cm | tiled GeoTIFF ZSTD | 1.26 MiB | 0.0289 m | 0.0498 m |
| uint16, 25 cm | tiled GeoTIFF ZSTD | 0.78 MiB | 0.0721 m | 0.1245 m |
| uint16, 50 cm | tiled GeoTIFF ZSTD | 0.54 MiB | 0.1443 m | 0.2490 m |
| uint16, 1 m | tiled GeoTIFF ZSTD | 0.37 MiB | 0.2887 m | 0.4981 m |

## Mission-Side Recommendation

For a mission terrain map that should preserve practically useful elevation detail without carrying unnecessary precision, use a tiled GeoTIFF or equivalent tiled binary raster:

| Setting | Recommendation |
|---|---|
| Raster shape | 2000 x 2000 |
| Horizontal spacing | 1.000500250125 m |
| Stored type | uint16 or int16 |
| Vertical scale | 0.10 m |
| Vertical offset | -42.17120622568093 m |
| Compression | ZSTD with horizontal predictor, or another delta/predictor-aware compressor |
| Expected size for this DEM | 1.26 MiB |
| Expected error | 2.9 cm RMSE, 5.0 cm max absolute error |

This is a strong default because the quantization error is far below the current NEMo GT DEM error scale and below the level that would matter for the path-planning experiments described here. If the mission requires more conservative archival quality, use 1 cm or 5 cm quantization. If the terrain map is only for coarse navigation, 25 cm quantization compresses below 1 MiB while keeping max height error near 12.5 cm.

## Practical Takeaways

- The full XYZ float64 DEM point cloud is not a storage-efficient mission format. It costs 91.6 MiB because x and y are repeated for every cell.
- A float32 height raster is already a near-lossless replacement at 15.3 MiB raw or 5.05 MiB compressed GeoTIFF.
- A 10 cm quantized raster is the best size/quality tradeoff from these measurements: 1.26 MiB compressed with centimeter-scale error.
- The NEMo height-field model is much smaller than any high-resolution raster: about 103 KiB on disk. That compactness is useful, but it is a learned approximation rather than a direct terrain product.
- Storage pressure in the current output tree comes from repeated Nerfstudio checkpoints and rendered PNGs, not from NEMo height-field weights.
