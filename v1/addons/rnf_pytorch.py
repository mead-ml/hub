import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from baseline.pytorch.classify import EmbedPoolStackClassifier
from baseline.model import register_model
from eight_mile.pytorch.layers import WithDropout

def ngrams(x, filtsz, mxlen):
    chunks = []
    for i in range(mxlen - filtsz + 1):
        chunk = x[:, i:i+filtsz, :]
        chunks += [chunk]
    chunks = torch.stack(chunks, 1)
    return chunks

class RNF(nn.Module):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.filtsz = kwargs['filtsz']
        self.mxlen = kwargs.get('mxlen', 100)
        pdrop = kwargs.get('dropout', 0.4)
        self.output_dim = kwargs.get('rnnsz', 300)
        self.rnf = nn.LSTM(input_dim, self.output_dim, batch_first=True)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        btc, lengths = inputs
        btc = self.dropout(btc)
        mxlen = max(torch.max(lengths), self.filtsz+1)
        btfc = ngrams(btc, self.filtsz, mxlen)
        B, T, F, C = btfc.shape
        btc = btfc.view(B*T, F, C)
        output, hidden = self.rnf(btc)
        hidden = self.dropout(hidden[0])
        hidden = hidden.view(hidden.shape[1:])
        btc = hidden.view(B, T, -1)
        bc = btc.max(1)[0]
        return bc

@register_model(task='classify', name='rnf')
class RNFWordClassifier(EmbedPoolStackClassifier):

    def init_pool(self, input_dim: int, **kwargs) -> nn.Module:
        return WithDropout(RNF(input_dim, **kwargs), kwargs.get('pool_dropout', 0.0))

