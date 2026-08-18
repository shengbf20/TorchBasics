import torch
from torch import nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    """
    def __repr__(self):
        return "This is my network!"
    """

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return nn.functional.softmax(logits, dim=1) # 将 logits 转换为概率


network = Network().to(device)
print(network)

print("--------------------------------")
X = torch.rand(1, 28, 28, device=device)
pred_probs = network(X)
pred_labels = pred_probs.argmax(dim=1)
print(f"Predicted class: {pred_labels}")



print("--------------------------------")
print(f"Model structure: {network}\n\n")

for name, param in network.named_parameters():
    print(f"Layer: {name} \nSize: {param.size()} \nNumel: {param.numel()} \nValues : {param[:2]} \n")