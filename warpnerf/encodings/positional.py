import torch

class PositionalEncoding(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        max_freq_log2: int,
        num_freq_bands: int,
        include_input: bool = True,
    ):
        
        super(PositionalEncoding, self).__init__()

        self.input_dim = input_dim
        self.max_freq_log2 = max_freq_log2
        self.num_freq_bands = num_freq_bands
        self.include_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
            
        if x.shape[0] == 0:
            return torch.zeros([0, self.output_dim], device=x.device)
        
        freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, self.num_freq_bands, device=x.device)

        out = []
        for freq_band in freq_bands:
            out.append(torch.sin(freq_band * x))
            out.append(torch.cos(freq_band * x))

        if self.include_input:
            out.append(x)

        x = torch.cat(out, dim=-1)

        return x
    
    @property
    def output_dim(self) -> int:
        return self.input_dim * (2 * self.num_freq_bands + (1 if self.include_input else 0))
   
    # @property
    # def input_dim(self) -> int:
    #     return self.input_dim
    