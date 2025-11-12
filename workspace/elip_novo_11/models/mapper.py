import torch
class MLPMapper(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim * num_tokens)
        )
    def forward(self, x):
        b = x.size(0)
        y = self.net(x)
        y = y.view(b, self.num_tokens, -1)
        return y
