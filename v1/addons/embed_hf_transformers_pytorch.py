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
from transformers import *
from baseline.pytorch.torchy import *
import copy
import json
import math
import re

MODELS = {"bert": (BertModel,       BertTokenizer,       BertConfig),
          "todbert": (BertModel,       BertTokenizer,       BertConfig),
          "gpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "todgpt2": (GPT2Model,       GPT2Tokenizer,       GPT2Config),
          "dialogpt": (AutoModelWithLMHead, AutoTokenizer, GPT2Config),
          "albert": (AlbertModel, AlbertTokenizer, AlbertConfig),
          "roberta": (RobertaModel, RobertaTokenizer, RobertaConfig),
          "distilbert": (DistilBertModel, DistilBertTokenizer, DistilBertConfig),
          "electra": (ElectraModel, ElectraTokenizer, ElectraConfig)}

class HFBaseEmbeddings(PyTorchEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dsz = kwargs.get('dsz')
        ModelType, TokenizerType, ConfigType = MODELS[kwargs.get('model_type', 'bert')]
        self.model = ModelType.from_pretrained(kwargs['handle'], cache_dir=kwargs.get('cache_dir', os.path.expanduser('~/.bl-data')))
        self.config = ConfigType.from_pretrained(kwargs['handle'])
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

    def forward(self, x):

        input_mask = torch.zeros(x.shape, device=x.device, dtype=torch.long).masked_fill(x != 0, 1)
        input_type_ids = torch.zeros(x.shape, device=x.device, dtype=torch.long)
        all_layers, pooled = self.model(x, token_type_ids=input_type_ids, attention_mask=input_mask)
        z = self.get_output(all_layers, pooled)
        return z

    def get_output(self, all_layers, pooled):
        pass


@register_embeddings(name='hf-embed')
class HFEmbeddings(HFBaseEmbeddings):

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


@register_embeddings(name='hf-pooled')
class HFPooledEmbeddings(HFBaseEmbeddings):

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

    def get_output(self, all_layers, pooled):
        return pooled

