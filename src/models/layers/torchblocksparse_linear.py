import torch
import torch.nn as nn

from pytorch_block_sparse import BlockSparseLinear

class TorchBlocksparseLinear(nn.Module):
    def __init__(self, in_features, out_features, density=0.5, bias=True):
        super(TorchBlocksparseLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.density = density

        self.module = BlockSparseLinear(self.in_features, self.out_features, density=self.density)

    @property
    def saving(self):
        return ((self.module.weight.numel())
                / (self.in_features * self.out_features))

    def forward(self, x):
        return self.module(x)