


#### [ELMo pre-trained embeddings](embed_elmo_tf.py)

- requires [tensorflow_hub](https://www.tensorflow.org/hub/installation)
  - `pip install tensorflow-hub`

This addon provides TensorFlow embeddings for pre-trained ELMo models.
There are 2 flavors provided, the TF-Hub version, and the local graph version

- The `elmo-embed` handler is a from-scratch local graph based on the original Allen repository's TensorFlow source code and it can load the HDF checkpoints provided by AllenAI directly.  Its input is a `[B, T]` temporal vector and its output is a `[B, T, H]` vector.  This the preferred handler for ELMo when you need to obtain per-word contextual embeddings

- The `elmo-pooled` handler is sub-class of the `elmo-embed` class and provides several forms of reduction to collapse the input tensor from `[B, T]` to `[B, H]` including:
  - Max over time (`pooling=max`)
  - Sum over time (`pooling=sum`)
  - Kim Convolution max-over-time pooling (`pooling=conv,filtsz=[3, 4, 5]`)
  - Defaults to mean over time

- The `elmo` handler provides an embedding of a temporal `string` vector `[B, T]`
  - because this model requires string input, it should be used with the `vec_text.py` addon and, in general, should be avoided where possible
  - The peformance of this model is listed under the [tagger section](../../docs/tagger.md)

## RNF classifier

- [RNF classifier](rnf_pytorch.py)
  - This is a PyTorch reimplemenation of the paper [Convolutional Neural Networks with Recurrent Neural Filters](https://www.groundai.com/project/convolutional-neural-networks-with-recurrent-neural-filters/) by Yi Yang
  - The original paper is here: https://arxiv.org/abs/1808.09315
  - The original Keras code is here: https://github.com/bloomberg/cnn-rnf
