from __future__ import annotations

from typing import Any

from torch import Tensor, nn

try:
    import tinycudann as tcnn
except ImportError:  # pragma: no cover - optional dependency
    tcnn = None


class TCNNHashGridColorDecoder(nn.Module):
    """Hash-grid RGB decoder over zero-to-one normalized XY coordinates."""

    def __init__(
        self,
        *,
        encoding_config: dict[str, Any] | None = None,
        network_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if tcnn is None:  # pragma: no cover - depends on external package
            raise ImportError(
                "tinycudann is not installed. Install the optional `nemo[tcnn]` extras."
            )
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
        self.encoding_config = dict(encoding_config)
        self.network_config = dict(network_config)
        self.encoding = tcnn.Encoding(n_input_dims=2, encoding_config=encoding_config)
        self.network = tcnn.Network(
            n_input_dims=self.encoding.n_output_dims,
            n_output_dims=3,
            network_config=network_config,
        )

    def forward(self, xy_zero_to_one: Tensor) -> Tensor:
        encoded = self.encoding(xy_zero_to_one)
        rgb_logits = self.network(encoded)
        return rgb_logits.sigmoid()
