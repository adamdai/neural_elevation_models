from __future__ import annotations

from typing import Any

from torch import Tensor

from nemo.height_field import Bounds, HeightField

try:
    import tinycudann as tcnn
except ImportError:  # pragma: no cover - optional dependency
    tcnn = None


class TCNNHashGridHeightField(HeightField):
    """Hash-grid encoded height field backed by tiny-cuda-nn."""

    def __init__(
        self,
        bounds: Bounds,
        *,
        encoding_config: dict[str, Any] | None = None,
        network_config: dict[str, Any] | None = None,
    ) -> None:
        if tcnn is None:  # pragma: no cover - depends on external package
            raise ImportError(
                "tinycudann is not installed. Install the optional `nemo[tcnn]` extras."
            )
        super().__init__(bounds)
        encoding_config = encoding_config or {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.5,
        }
        network_config = network_config or {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
        self.encoding = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config)
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=1,
            network_config=network_config,
        )

    def h(self, xy: Tensor) -> Tensor:
        xy_norm = self.normalizer.normalize_zero_to_one(xy)
        encoded = self.encoding(xy_norm)
        return self.network(encoded)
