import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(3, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

x = torch.tensor([
    [float('nan'), 1.0, 2.0],
    [0.0, float('nan'), 2.0],
    [1.0, 2.0, float('nan')]
], requires_grad=True)

y = torch.tensor([[1.0], [2.0], [3.0]])
mask = ~torch.isnan(x).any(dim=1)

model = TinyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("\n--- NaN-affected training ---")
out = model(x)
if mask.any():
    loss = F.mse_loss(out[mask], y[mask])
    loss.backward()
else:
    print("✅ Loss skipped — all rows are masked")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad has NaN? {torch.isnan(param.grad).any().item()}")

print("\n--- FIXED VERSION ---")
x_clean = torch.nan_to_num(x, nan=0.0)
model = TinyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
out = model(x_clean)
if mask.any():
    loss = F.mse_loss(out[mask], y[mask])
    loss.backward()
else:
    print("✅ Loss skipped — all rows are masked")
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad has NaN? {torch.isnan(param.grad).any().item()}")
