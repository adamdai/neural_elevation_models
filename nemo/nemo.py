import torch
import tinycudann as tcnn

from nerfstudio.field_components.spatial_distortions import SceneContraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: should inherit from nn.Module?
class Nemo:
    def __init__(self, encs_pth=None, mlp_pth=None):
        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.2599210739135742,
            },
        )
        tot_out_dims_2d = 128

        self.heightcap_net = tcnn.Network(
            n_input_dims=tot_out_dims_2d,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 1,
            },
        )

        self.spatial_distortion = SceneContraction()

        if encs_pth is not None and mlp_pth is not None:
            self.load_weights(encs_pth, mlp_pth)

        # TODO: add xyz scaling
        self.x_scale = None
        self.y_scale = None
        self.z_scale = None


    def load_weights(self, encs_pth, mlp_pth):
        self.encoding.load_state_dict(torch.load(encs_pth))
        self.encoding.to(device)
        self.heightcap_net.load_state_dict(torch.load(mlp_pth))
        self.heightcap_net.to(device)


    def get_heights(self, positions):
        positions = torch.cat([positions, torch.zeros_like(positions[..., :1])], dim=-1)
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        with torch.no_grad():
            encs = self.encoding(positions[:, :2])
            heights = self.heightcap_net(encs)
        return heights
    
    def get_heights_with_grad(self, positions):
        positions = torch.cat([positions, torch.zeros_like(positions[..., :1])], dim=-1)
        positions = self.spatial_distortion(positions)
        positions = (positions + 2.0) / 4.0
        encs = self.encoding(positions[:, :2])
        heights = self.heightcap_net(encs)
        grad = torch.autograd.grad(heights.sum(), positions, create_graph=True)[0]
        return heights, grad