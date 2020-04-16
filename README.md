# mead-hub
A hub for pre-trained [MEAD 2.0](https://github.com/dpressel/mead-baseline/tree/feature/v2) models.

This is inspired by hubs provided by the TensorFlow and PyTorch communities.  The objective here is to share models and model configurations that work inside MEAD to train and evaluate Deep Learning-based NLP.  Prior to this hub, most addons were provided either directly in the core repository or in private repositories causing a proliferation of custom embedding and vectorizer indices and no canonical way to reference these models from the MEAD config files.

Now that MEAD supports direct download of indices, configurations, Embeddings and any code addons, creating a central repository of 3rd party contributions makes a lot of sense.  Additionally, for hub modules on the master branch, there is a canonical shorthand way of referencing the indices provided here which makes things even easier

## Hub Types

This repository supports several types of mead-specific models including:
- Any python modules that override registered classes and are brought in by the mead configuration files in mead-train
- python task_modules, which are custom defined modules providing a subclass to Task that can be passed into mead-train to allow it train new tasks
- vectorizer indices that are provided to mead-train to allow it to find and download a specific vectorizer and use it for training
- embeddings indices that are provided to mead-train to allow it to find and download model checkpoints and refer to any required addons needed for mead-train to load them

