import collections
import unicodedata
import six
import numpy as np
from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddings
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, load_bert_vocab
from transformers import AutoTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import AutoModel
from baseline.pytorch.torchy import *
import copy
import json
import math
import re

class HFTransformersEmbeddings(PyTorchEmbeddings):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.handle = kwargs.get('handle')
        self.dsz = kwargs.get('dsz', 768)
        self.d_model = kwargs.get('d_model', self.dsz)
        self.add_pooling_layer = bool(kwargs.get('add_pooling_layer', False))
        self.model = AutoModel.from_pretrained(kwargs.get('handle'), add_pooling_layer=self.add_pooling_layer)
        tok = AutoTokenizer.from_pretrained(kwargs.get('handle'))
        self.vocab = tok.get_vocab()
        self.PAD = tok.pad_token_id if hasattr(tok, 'pad_token_id') else 0
        self.CLS = tok.cls_token_id
        self.vsz = len(self.vocab)

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("roberta", **kwargs)
        c.checkpoint = embeddings
        return c

    def forward(self, input_ids, _=None):
        input_mask = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.long)\
            .masked_fill(input_ids != self.PAD, 1)
        output = self.model(input_ids, attention_mask=input_mask,
                            output_hidden_states=True)
        z = self.get_output(input_ids, output)
        return z

    def get_output(self, input_ids, output):
        pass


@register_embeddings(name='hf-transformers-embed')
class HFTransformersEmbeddings(HFTransformersEmbeddings):

    def __init__(self, name, **kwargs):
        """HFTransformerw sequence embeddings, used for a feature-ful representation of finetuning sequence tasks.

        If operator == 'concat' result is [B, T, #Layers * H] other size the layers are mean'd the shape is [B, T, H]
        """
        super().__init__(name=name, **kwargs)
        self.layer_indices = kwargs.get('layers', [-1])
        self.operator = kwargs.get('operator', 'concat')
        self.finetune = kwargs.get('trainable', kwargs.get('finetune', False))

    def get_output(self, input_ids, output):
        all_layers = output[-1]
        if self.finetune:
            layers = [all_layers[layer_index] for layer_index in self.layer_indices]
        else:
            layers = [all_layers[layer_index].detach() for layer_index in self.layer_indices]
        if self.operator != 'concat':
            z = torch.cat([l.unsqueeze(-1) for l in layers], dim=-1)
            z = torch.mean(z, dim=-1)
        else:
            z = torch.cat(layers, dim=-1)
        return z

    def extra_repr(self):
        return f"finetune={self.finetune}, combination={self.operator}, layers={self.layer_indices}"


@register_embeddings(name='hf-transformers-pooled')
class HFTransformersPooledEmbeddings(HFTransformersEmbeddings):

    def __init__(self, name, pooling='cls', **kwargs):
        self.pooling = pooling
        kwargs['add_pooling_layer'] = bool(kwargs.get('add_pooling_layer', self.pooling == 'default'))
        super().__init__(name=name, **kwargs)
        
        
        if self.pooling == 'default':
            self.pooling_op = self._pooler
        elif self.pooling == 'cls':
            self.pooling_op = self._cls_pool
        elif self.pooling == 'mean':
            self.pooling_op = self._mean_pool
        elif self.pooling == 'max':
            self.pooling_op = self._max_pool
        else:
            raise Exception(f"Unknown pooling type {self.pooling}")
            

    def _pooler(self, inputs, output):
        return output[1]
        
    def get_dsz(self):
        return self.d_model

    def _mean_pool(self, inputs, output):
        embeddings = output[0]
        mask = (inputs != self.PAD)
        seq_lengths = mask.sum(1).float()
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
        return embeddings.sum(1)/seq_lengths.unsqueeze(-1)

    def _max_pool(self, inputs, output):
        embeddings = output[0]
        mask = (inputs != self.PAD)
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1) == False, 0.)
        return torch.max(embeddings, 1, False)[0]

    def _cls_pool(self, inputs, output):
        tensor = output[0]
        B = tensor.shape[0]
        mask = (inputs == self.CLS).unsqueeze(-1).expand_as(tensor)
        pooled = tensor.masked_select(mask).view(B, -1)
        return pooled

    def get_output(self, inputs, z):
        z = self.pooling_op(inputs, z)
        return z
