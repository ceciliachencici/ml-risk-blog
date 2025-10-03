import torch
import torch.nn as nn
import torch.nn.functional as F

# Sample input: 3 sequences with NaNs due to padding
X = torch.tensor([
    [1.0, 2.0, 3.0, float('nan'), float('nan')],
    [5.0, float('nan'), float('nan'), float('nan'), float('nan')],
    [1.0, 2.0, 3.0, 4.0, 5.0]
])
mask = ~torch.isnan(X)

class PositionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(5, dim))

    def forward(self, x):
        pos = self.weight[:x.shape[1]]
        return x + pos

class RiskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 8)
        self.pos = PositionEncoder(8)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, T, 1]
        x = self.embedding(x)
        x = self.pos(x)
        x = self.head(x)
        return x.squeeze(-1)

model = RiskEncoder()
X.requires_grad = True
out = model(X)  # NaNs propagate
y = torch.ones_like(out)
loss = ((out - y)**2)[mask].mean()
loss.backward()

for name, param in model.named_parameters():
    print(f"{name} grad has NaN? {torch.isnan(param.grad).any().item()}")
