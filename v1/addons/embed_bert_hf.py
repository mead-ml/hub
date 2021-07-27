from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import unicodedata
import six
import numpy as np
from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddings
from baseline.vectorizers import register_vectorizer, AbstractVectorizer, load_bert_vocab
from transformers import BertTokenizer
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from transformers import BertModel
from baseline.pytorch.torchy import *
import copy
import json
import math
import re


class BERTBaseEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dsz = kwargs.get('dsz')
        self.model = BertModel.from_pretrained(kwargs.get('handle'))
        self.vocab = load_bert_vocab(None)
        self.vsz = len(self.vocab)  # 30522 self.model.embeddings.word_embeddings.num_embeddings

    def get_vocab(self):
        return self.vocab

    def get_dsz(self):
        return self.dsz

    def get_vsz(self):
        return self.vsz

    @classmethod
    def load(cls, embeddings, **kwargs):
        c = cls("bert", **kwargs)
        c.checkpoint = embeddings
        return c

    def forward(self, input_ids, token_type_ids=None):
        input_mask = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.long)\
            .masked_fill(input_ids != 0, 1)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.shape, device=input_ids.device, dtype=torch.long)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask,
                            output_hidden_states=True)
        # last_layer: BxTxh, pooled: Bxh, all_layers = num_layers_encoder*BxTxh
        last_layer, pooled, all_layers = output[0], output[1], output[2]
        z = self.get_output(all_layers, pooled)
        return z

    def get_output(self, all_layers, pooled):
        pass


@register_embeddings(name='bert')
class BERTEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        """BERT sequence embeddings, used for a feature-ful representation of finetuning sequence tasks.

        If operator == 'concat' result is [B, T, #Layers * H] other size the layers are mean'd the shape is [B, T, H]
        """
        super().__init__(name=name, **kwargs)
        self.layer_indices = kwargs.get('layers', [-1, -2, -3, -4])
        self.operator = kwargs.get('operator', 'concat')
        self.finetune = kwargs.get('trainable', kwargs.get('finetune', False))

    def get_output(self, all_layers, pooled):
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


@register_embeddings(name='bert-pooled')
class BERTPooledEmbeddings(BERTBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def get_output(self, all_layers, pooled):
        return pooled
