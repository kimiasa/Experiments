import torch
import torch.nn as nn

class LowrankLinear(nn.Module):
    def __init__(self, in_features, out_features, redn_factor=1.0, bias=True):
        super(LowrankLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.intermediate_dim = int((in_features * out_features) / ((in_features + out_features) * redn_factor))

        assert(self.intermediate_dim > 0)

        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.zeros(
                out_features, dtype=torch.float), requires_grad=True)

        self.module = nn.Sequential(
            nn.Linear(in_features, self.intermediate_dim, bias=False),  # q_right, no bias between q_right and q_left
            # No activation between q_right and q_left
            nn.Linear(self.intermediate_dim, out_features, bias=bias),  # q_left
        )

    @property
    def saving(self):
        return (((self.in_features * self.intermediate_dim) + (self.out_features * self.intermediate_dim))
                / (self.in_features * self.out_features))

    def forward(self, x):
        return self.module(x)