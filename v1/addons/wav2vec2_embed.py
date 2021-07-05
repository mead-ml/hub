from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddingsModel
from eight_mile.pytorch.embeddings import PyTorchEmbeddings
from eight_mile.pytorch.layers import sequence_mask_mxlen
import torch
from audio8.wav2vec2 import Wav2Vec2Encoder, Wav2Vec2PooledEncoder, CONV_FEATURES
import numpy as np
import os
from eight_mile.pytorch.layers import EmbeddingsStack
from baseline.pytorch import TensorDef, BaseLayer
import time
import soundfile as sf


class Wav2Vec2PooledEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__()
        reduction_type = kwargs.get('reduction_type', 'max')
        sample_rate = kwargs.get('sample_rate', 8)
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        self.encoder = Wav2Vec2PooledEncoder(conv_features=CONV_FEATURES[sample_rate], d_model=self.d_model, reduction_type=reduction_type)
        self.unfreeze_after = int(kwargs.get('unfreeze_after', 50_000))
        self.steps = 0

    def get_vsz(self):
        return 0

    def get_dsz(self):
        return self.d_model

    def get_vocab(self):
        return {}

    @property
    def dsz(self):
        return self.d_model

    def forward(self, x, pad_mask=None):
        if self.encoder.freeze and self.steps > self.unfreeze_after:
            print('Unfreezing encoder')
            self.encoder.freeze = False
        z = self.encoder((x, pad_mask))
        self.steps += 1
        return z

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls(**kwargs)
        mapping = torch.load(embeddings)
        print(c.encoder.load_state_dict(mapping, strict=False))
        return c


@register_embeddings(name='wav2vec2-pooled')
class Wav2Vec2PooledEmbeddingsModel(PyTorchEmbeddingsModel, Wav2Vec2PooledEmbeddings):
    """Register embedding model for usage in mead"""
    pass
