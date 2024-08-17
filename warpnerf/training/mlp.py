import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int, activation=nn.ReLU):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = activation()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)

        return self.output_layer(x)
