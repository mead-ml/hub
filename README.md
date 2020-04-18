# mead-hub
A hub for pre-trained [MEAD 2.0](https://github.com/dpressel/mead-baseline/tree/feature/v2) models.

This is inspired by hubs provided by the TensorFlow and PyTorch communities.  The objective here is to share models and model configurations that work inside MEAD to train and evaluate Deep Learning-based NLP.  Prior to this hub, most addons were provided either directly in the core repository or in private repositories causing a proliferation of custom embedding and vectorizer indices and no canonical way to reference these models from the MEAD config files.

Now that MEAD supports direct download of indices, configurations, Embeddings and any code addons, creating a central repository of 3rd party contributions makes a lot of sense.  Additionally, for hub modules on the master branch, there is a canonical shorthand way of referencing the indices provided here which makes things even easier

## Hub Types

This repository supports several types of mead-specific models including:
- Any python modules that override registered classes and are brought in by the mead configuration files in `mead-train`
- python `task_modules`, which are custom defined modules providing a subclass to `Task` that can be passed into mead-train to allow it train new tasks
- vectorizer indices that are provided to mead-train to allow it to find and download a specific vectorizer and use it for training
- embeddings indices that are provided to mead-train to allow it to find and download model checkpoints and refer to any required addons needed for mead-train to load them

## Using and referencing Hub from MEAD

Here is an example call to `mead-train` to fine-tune BERT for classification on SST2:

```
mead-train --embeddings hub:v1:embeddings --config config/sst2-bert-base-uncased.yml --vecs hub:v1:vecs
```

The optional index arguments `--embeddings` and `--vecs` have been supplied here as shortname references to mead-hub.  This causes `mead-train` to download these indices and allows us to reference the labels from those indices (which are usually referencs to hub addons).  Here is what the mead config looks like in its entirety:

```
backend: pytorch
basedir: ./bert-base-uncased-sst2
batchsz: 12
dataset: SST2
modules:
- embed_bert
features:
- embeddings:
    label: bert-base-uncased-pytorch
    type: bert-embed-pooled
    finetune: true
    dropout: 0.1

  name: bert
  vectorizer:
    label: bert-base-uncased
loader:
  reader_type: default
model:
  model_type: fine-tune
task: classify
train:
  early_stopping_metric: acc
  epochs: 5
  eta: 4.0e-5
  optim: adamw
  weight_decay: 1.0e-3

```

The SST2 dataset is defined in the default datasets index (note that we didnt override that), but notice that the embeddings label (`bert-base-uncased-pytorch`) and the vectorizer label (`bert-base-uncased`) reference items in the mead hub embeddings and vecs indices.  That's it!  The hub indices are downloaded, and when referenced in the mead config, the appropriate addons are automatically downloaded to the mead-baseline cache!


