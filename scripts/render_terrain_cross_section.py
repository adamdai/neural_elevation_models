from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tyro


CUSTOM_COLORMAPS = {
    "purple_blue_teal": LinearSegmentedColormap.from_list(
        "purple_blue_teal",
        [
            "#481567",
            "#3f4788",
            "#287d8e",
            "#20a486",
        ],
    )
}


@dataclass
class TerrainCrossSectionArgs:
    output_path: str = "outputs/terrain_cross_section/terrain_cross_section.png"
    width_px: int = 4000
    height_px: int = 1400
    dpi: int = 300
    seed: int = 12
    colormap: str = "purple_blue_teal"
    colormap_min: float = 0.0
    colormap_max: float = 1.0
    min_elevation: float = 0.0
    max_elevation: float = 100.0
    base_elevation: float = 6.0
    mean_surface_elevation: float = 58.0
    relief: float = 34.0
    broad_bump_weight: float = 1.0
    random_walk_smoothing_px: float = 75.0
    wave_bump_weight: float = 0.22
    detail_smoothing_px: float = 17.0
    detail_weight: float = 0.24
    profile_points: int = 900
    margin_px: int = 42
    transparent_background: bool = True
    background: tuple[float, float, float] = (0.96, 0.97, 0.96)
    outline: bool = False
    outline_width_px: int = 5
    outline_color: tuple[float, float, float] = (0.10, 0.10, 0.10)
    vertical_exaggeration: float = 1.0


def main(args: TerrainCrossSectionArgs) -> None:
    if args.width_px <= 0 or args.height_px <= 0:
        raise ValueError("width_px and height_px must be positive")
    if args.profile_points < 8:
        raise ValueError("profile_points must be at least 8")
    if args.max_elevation <= args.min_elevation:
        raise ValueError("max_elevation must be greater than min_elevation")
    if args.base_elevation < args.min_elevation:
        raise ValueError("base_elevation must be at least min_elevation")
    if not 0.0 <= args.colormap_min <= 1.0:
        raise ValueError("colormap_min must be in [0, 1]")
    if not 0.0 <= args.colormap_max <= 1.0:
        raise ValueError("colormap_max must be in [0, 1]")
    if args.colormap_max < args.colormap_min:
        raise ValueError("colormap_max must be greater than or equal to colormap_min")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    surface = make_bumpy_profile(
        points=int(args.profile_points),
        seed=int(args.seed),
        mean_elevation=float(args.mean_surface_elevation),
        relief=float(args.relief),
        min_elevation=float(args.base_elevation) + 0.08 * float(args.relief),
        max_elevation=float(args.max_elevation),
        broad_bump_weight=float(args.broad_bump_weight),
        random_walk_smoothing_px=float(args.random_walk_smoothing_px),
        wave_bump_weight=float(args.wave_bump_weight),
        detail_smoothing_px=float(args.detail_smoothing_px),
        detail_weight=float(args.detail_weight),
    )
    image = render_cross_section(surface, args)

    plt.imsave(output_path, image, dpi=int(args.dpi))
    print(output_path)


def make_bumpy_profile(
    *,
    points: int,
    seed: int,
    mean_elevation: float,
    relief: float,
    min_elevation: float,
    max_elevation: float,
    broad_bump_weight: float,
    random_walk_smoothing_px: float,
    wave_bump_weight: float,
    detail_smoothing_px: float,
    detail_weight: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    walk = np.cumsum(rng.normal(0.0, 1.0, size=points))
    walk = smooth_1d(walk, sigma_px=random_walk_smoothing_px)
    detail = smooth_1d(rng.normal(0.0, 1.0, size=points), sigma_px=detail_smoothing_px)

    terrain = float(broad_bump_weight) * _standardize(walk)
    terrain += float(detail_weight) * _standardize(detail)
    terrain = _standardize(terrain)
    terrain = mean_elevation + 0.5 * relief * terrain

    # Add broad deterministic undulations so the silhouette reads well as terrain.
    x = np.linspace(0.0, 1.0, points)
    terrain += float(wave_bump_weight) * 0.55 * relief * np.sin(2.0 * np.pi * (1.5 * x + 0.08))
    terrain += float(wave_bump_weight) * 0.32 * relief * np.sin(2.0 * np.pi * (3.3 * x + 0.31))

    return np.clip(terrain, min_elevation, max_elevation)


def render_cross_section(surface: np.ndarray, args: TerrainCrossSectionArgs) -> np.ndarray:
    height = int(args.height_px)
    width = int(args.width_px)
    margin = max(0, int(args.margin_px))
    plot_width = max(2, width - 2 * margin)
    plot_height = max(2, height - 2 * margin)

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    if not args.transparent_background:
        rgba[..., :3] = np.array(args.background, dtype=np.float32)
        rgba[..., 3] = 1.0

    x_src = np.linspace(0.0, 1.0, surface.size)
    x_dst = np.linspace(0.0, 1.0, plot_width)
    surface_dst = np.interp(x_dst, x_src, surface)
    surface_dst = _exaggerate_about_base(
        surface_dst,
        base=float(args.base_elevation),
        factor=float(args.vertical_exaggeration),
    )
    surface_dst = np.clip(surface_dst, args.min_elevation, args.max_elevation)

    cmap = get_colormap(args.colormap)
    y_norm = np.linspace(1.0, 0.0, plot_height, dtype=np.float32)
    y_elevation = args.min_elevation + y_norm * (args.max_elevation - args.min_elevation)
    surface_max = float(np.max(surface_dst))
    terrain_min = float(args.base_elevation)
    terrain_span = max(surface_max - terrain_min, 1e-6)
    terrain_relative_height = np.clip((y_elevation - terrain_min) / terrain_span, 0.0, 1.0)
    cmap_values = args.colormap_min + terrain_relative_height * (args.colormap_max - args.colormap_min)
    vertical_color = np.asarray(cmap(cmap_values), dtype=np.float32)

    row_elevation = y_elevation[:, None]
    inside = (row_elevation <= surface_dst[None, :]) & (row_elevation >= float(args.base_elevation))
    block = rgba[margin : margin + plot_height, margin : margin + plot_width]
    block[inside] = vertical_color[:, None, :].repeat(plot_width, axis=1)[inside]

    if args.outline:
        draw_surface_outline(
            rgba,
            surface_dst,
            args,
            margin=margin,
            plot_height=plot_height,
            plot_width=plot_width,
        )

    return rgba


def draw_surface_outline(
    rgba: np.ndarray,
    surface: np.ndarray,
    args: TerrainCrossSectionArgs,
    *,
    margin: int,
    plot_height: int,
    plot_width: int,
) -> None:
    color = np.array((*args.outline_color, 1.0), dtype=np.float32)
    thickness = max(1, int(args.outline_width_px))
    denom = float(args.max_elevation - args.min_elevation)
    y = (args.max_elevation - surface) / denom * float(plot_height - 1)
    y = np.rint(y).astype(np.int32) + margin

    for x, center_y in enumerate(y):
        image_x = x + margin
        y0 = max(margin, center_y - thickness // 2)
        y1 = min(margin + plot_height, center_y + (thickness + 1) // 2)
        rgba[y0:y1, image_x] = color

    base_y = (
        int(round((args.max_elevation - args.base_elevation) / denom * float(plot_height - 1)))
        + margin
    )
    y0 = max(margin, base_y - thickness // 2)
    y1 = min(margin + plot_height, base_y + (thickness + 1) // 2)
    rgba[y0:y1, margin : margin + plot_width] = color


def smooth_1d(values: np.ndarray, *, sigma_px: float) -> np.ndarray:
    sigma = max(0.0, float(sigma_px))
    if sigma == 0.0:
        return values.astype(np.float64, copy=True)

    radius = max(1, int(round(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    padded = np.pad(values.astype(np.float64), radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def get_colormap(name: str):
    if name in CUSTOM_COLORMAPS:
        return CUSTOM_COLORMAPS[name]
    return plt.get_cmap(name)


def _standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    std = float(np.std(values))
    if std < 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def _exaggerate_about_base(values: np.ndarray, *, base: float, factor: float) -> np.ndarray:
    return base + (values - base) * max(0.0, factor)


if __name__ == "__main__":
    main(tyro.cli(TerrainCrossSectionArgs))
