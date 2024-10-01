import torch
import tinycudann as tcnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeuralField(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoding = tcnn.Encoding(
            n_input_dims=2,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 8,
                "n_features_per_level": 8,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.2599210739135742,
                "interpolation": "Smoothstep"
            },
        )
        tot_out_dims_2d = self.encoding.n_output_dims

        self.mlp = tcnn.Network(
            n_input_dims=tot_out_dims_2d,
            n_output_dims=1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 256,
                "n_hidden_layers": 3,
            },
        )

    def forward(self, x):
        x = self.encoding(x)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    nf = NeuralField()
    nf.to(device)

    x = torch.rand(10, 2, requires_grad=True).to(device)
    y = nf(x)

    # 2 methods of obtaining 2nd derivative - both give the same error
    method = 0

    if method == 0:
        grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        grad_2 = torch.autograd.grad(grad.sum(), x)[0]
    else:
        grad = torch.autograd.grad(y, x, torch.ones_like(y, device=x.device), 
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_2 = torch.autograd.grad(grad, x, torch.ones_like(grad, device=x.device), 
                                    create_graph=False, retain_graph=False, only_inputs=True)[0]