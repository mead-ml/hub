# mead-hub
A hub for pre-trained [MEAD 2.0](https://github.com/dpressel/mead-baseline/tree/feature/v2) models.

A hub to share models, addons, and model configurations that work inside MEAD to train and evaluate Deep Learning-based NLP.

## Hub Types

This repository supports several types of MEAD-specific models including:
- Any python modules that override registered classes and are brought in by the mead configuration files in `mead-train`
- python `task_modules`, which are custom defined modules providing a subclass to `Task` that can be passed into mead-train to allow it train new tasks
- vectorizer indices that are provided to `mead-train` to allow it to find and download a specific vectorizer and use it for training
- embeddings indices that are provided to `mead-train` to allow it to find and download model checkpoints and refer to any required addons needed for mead-train to load them

## Using and Referencing Hub from MEAD

The hub embeddings can be referenced by passing them as the embedding argument to `mead-train`: `--embeddings hub:v1:embeddings`.  The vectorizers can be similarly passed with `--vecs hub:v1:vecs`.

The optional index arguments `--embeddings` and `--vecs` have been supplied here as shortname references to mead-hub.  This causes `mead-train` to download these indices and allows us to reference the labels from those indices (which are usually referencs to hub addons).

The hub indices are downloaded, and when referenced in the MEAD config, the appropriate addons are automatically downloaded to the mead-baseline cache!


