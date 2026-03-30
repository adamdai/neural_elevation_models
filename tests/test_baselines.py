from __future__ import annotations

import torch

from nemo import fit_plane_baseline


def test_fit_plane_baseline_recovers_plane() -> None:
    xy = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    z = 2.0 * xy[:, :1] - 3.0 * xy[:, 1:2] + 0.5
    plane = fit_plane_baseline(xy, z)
    pred = plane(xy)
    assert torch.allclose(pred, z, atol=1e-6)
