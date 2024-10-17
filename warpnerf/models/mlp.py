import torch.nn as nn

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_hidden_layers: int,
            activation=nn.ReLU,
            output_activation=None
        ):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation()
        self.output_activation = output_activation() if output_activation is not None else None

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)

        x = self.output_layer(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        
        return x
