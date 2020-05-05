"""
Hierarchically-Refined Label Attention Network for Sequence Labeling from here https://arxiv.org/pdf/1908.08676.pdf

Couple notes of possible difference with the paper because it isn't clear

1. In section 4.2 when discussing Using multihead attention to compute H^t they have this equation

    H^l = concat(head, ..., head_k) + H^w

   This seems to suggest they have a residual connection around the multi head attention, this seems weird given that at
   the next step they also concatenate with H^w. Their code doesn't do this residual connection suggested by their paper
   https://github.com/Nealcly/BiLSTM-LAN/blob/082fb6aec69b468bcfb0bff5aeaa2e43f4073965/model/lstm_attention.py#L24

2. The paper doesn't say that the last layer only has a single attention head but it makes sense, other wise you would only get
   to assign the probability to a subset of labels given a subset of the features. In the code they set their last num_heads to 1.
   https://github.com/Nealcly/BiLSTM-LAN/blob/082fb6aec69b468bcfb0bff5aeaa2e43f4073965/model/wordsequence.py#L44

3. In their code they have a query masking step which seems pointless? You don't need to mask the padded inputs because
   their attentions are calculated separately from each other so even though they will calculate junk as long as the loss
   ignores them it doesn't matter.
"""

from typing import Optional
import tensorflow as tf
from eight_mile.tf.embeddings import LookupTableEmbeddings
from eight_mile.tf.layers import (
    BiLSTMEncoderSequence,
    MultiHeadedAttention,
    tensor_and_lengths,
    SequenceSequenceAttention,
    PassThru,
    get_shape_as_list,
)
from baseline.model import register_model
from baseline.tf.tagger.model import AbstractEncoderTaggerModel


class BiLSTMLANEncoder(tf.keras.layers.Layer):
    def __init__(self, insz, hsz, nlayers, nlabels, num_heads=4, name="blstm_lan_encoder", **kwargs):
        super().__init__(name=name)
        assert nlayers > 1, "You need at least 2 layers for a BiLSTMLANEncoder"
        blstm_dropout = kwargs.get('blstm_dropout', 0.5)
        self.blstms = [BiLSTMEncoderSequence(insz, hsz, 1, pdrop=blstm_dropout, name=f"blstm/0", **kwargs)]
        for i in range(nlayers - 1):
            self.blstms.append(
                BiLSTMEncoderSequence(hsz * 2, hsz, 1, pdrop=blstm_dropout, name=f"blstm/{i + 1}", **kwargs)
            )

        mha_dropout = kwargs.get('mha_dropout, 0.1')
        self.mhas = [
            MultiHeadedAttention(num_heads=num_heads, d_model=hsz, dropout=mha_dropout, scale=True, name=f"mha/{i}")
            for i in range(nlayers - 1)
        ]
        self.mhas.append(
            TruncatedMultiHeadedAttention(num_heads=1, d_model=hsz, dropout=0.0, scale=True, name=f"mha/{nlayers - 1}")
        )

        self.label_embed = LookupTableEmbeddings(vsz=nlabels, dsz=hsz, name="label_embeddings")
        self.nlabels = nlabels

    @property
    def output_dim(self):
        return self.nlabels

    def call(self, inputs):
        inputs, lengths = tensor_and_lengths(inputs)
        batchsz = tf.shape(inputs)[0]
        labels = self.label_embed(tf.range(self.nlabels))
        labels = tf.expand_dims(labels, 0)
        labels = tf.tile(labels, [batchsz, 1, 1])
        for blstm, mha in zip(self.blstms[:-1], self.mhas[:-1]):
            out = blstm((inputs, lengths))
            label_attn = mha((out, labels, labels, None))
            inputs = tf.concat([out, label_attn], axis=2)

        out = self.blstms[-1]((inputs, lengths))
        attn_weights = self.mhas[-1]((out, labels, labels, None))
        return attn_weights


@register_model(task='tagger', name='blstm-lan')
class BLSTMLANTaggerModel(AbstractEncoderTaggerModel):
    def init_encode(self, **kwargs):
        nlayers = int(kwargs.get('layers', 2))
        blstm_dropout = float(kwargs.get('blstm_dropout', 0.5))
        mha_dropout = float(kwargs.get('mha_dropout', 0.1))
        unif = kwargs.get('unif', 0)
        hsz = int(kwargs['hsz'])
        weight_init = kwargs.get('weight_init', 'uniform')
        num_heads = int(kwargs.get('num_heads', 4))
        return BiLSTMLANEncoder(
            None,
            hsz,
            nlayers,
            len(self.labels),
            num_heads=num_heads,
            blstm_dropout=blstm_dropout,
            mha_dropout=mha_dropout,
            unif=unif,
            weight_init=weight_init,
        )

    def init_proj(self, **kwargs):
        return PassThru(self.encoder.output_dim)


# The Scaled Dot product attention module calculates the softmax of of layer so that it can
# calculate a weighted sum of the values based on the attention scores. Currently mead isn't
# set up to allow picking loss functions for tasks so this clashes with the CrossEntropyLoss
# used by the tagger. These are special classes that skip doing that.
class TruncatedSeqScaledDotProductAttention(SequenceSequenceAttention):
    def __init__(self, pdrop=0.1, name="scaled_dot_product_attention", **kwargs):
        super().__init__(pdrop, name=name, **kwargs)

    def _attention(self, query, key, mask=None):
        scores = tf.matmul(query, key, transpose_b=True)
        scores *= tf.math.rsqrt(tf.cast(tf.shape(query)[2], tf.float32))

        if mask is not None:
            scores = masked_fill(scores, mask == 0, -1e9)
        return scores

    def _update(self, a, _):
        return a


# This is a partial MHA class that only returns the attention weights instead of combining it
# with the values. It also removes the unneeded value and output parameters
class TruncatedMultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        dropout: float = 0.1,
        scale: bool = False,
        d_k: Optional[int] = None,
        name: str = None,
    ):
        """Constructor for multi-headed attention

        :param h: The number of heads
        :param d_model: The model hidden size
        :param dropout (``float``): The amount of dropout to use
        :param attn_fn: A function to apply attention, defaults to SDP
        """
        super().__init__(name=name)

        if d_k is None:
            self.d_k = d_model // num_heads
            if d_model % num_heads != 0:
                raise Exception(f"d_model ({d_model}) must be evenly divisible by num_heads ({num_heads})")
        else:
            self.d_k = d_k

        self.h = num_heads
        self.w_Q = tf.keras.layers.Dense(units=self.d_k * self.h, name="query_projection")
        self.w_K = tf.keras.layers.Dense(units=self.d_k * self.h, name="key_projection")
        self.attn_fn = TruncatedSeqScaledDotProductAttention(dropout)

    def call(self, qkvm):
        query, key, value, mask = qkvm
        batchsz = get_shape_as_list(query)[0]

        # (B, T, H, D) -> (B, H, T, D)
        query = tf.transpose(tf.reshape(self.w_Q(query), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        key = tf.transpose(tf.reshape(self.w_K(key), [batchsz, -1, self.h, self.d_k]), [0, 2, 1, 3])
        x = self.attn_fn((query, key, value, mask))
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.squeeze(x, axis=2)
        return x
