import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from baseline.pytorch.classify import EmbedPoolStackClassifier
from baseline.model import register_model
from eight_mile.pytorch.layers import ConvEncoderStack, MeanPool1D, MaxPool1D

class ResidualConvPool(nn.Module):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        filtsz = kwargs['filtsz']
        dropout = kwargs.get('dropout', 0.1)
        hsz = kwargs.get('hsz', 300)
        activation = kwargs.get('activation', 'relu')
        nlayers = kwargs.get('layers', 3)
        self.convs = ConvEncoderStack(input_dim, hsz, filtsz, nlayers=nlayers, pdrop=dropout, activation=activation)
        self.output_dim = 2 * hsz
        self.mean_pool = MeanPool1D(hsz)
        self.max_pool = MaxPool1D(hsz)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, lengths = inputs
        outputs = self.convs(x)
        mx = self.max_pool(outputs)
        mu = self.mean_pool((outputs, lengths))
        output = torch.cat([mx, mu], -1)
        return output

@register_model(task='classify', name='resconv')
class ConvClassifier(EmbedPoolStackClassifier):

    def init_pool(self, input_dim: int, **kwargs) -> nn.Module:
        return ResidualConvPool(input_dim, **kwargs)

