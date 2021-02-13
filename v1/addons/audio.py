from baseline.embeddings import register_embeddings
from baseline.pytorch.embeddings import PyTorchEmbeddingsModel
from eight_mile.pytorch.embeddings import PyTorchEmbeddings
from eight_mile.pytorch.layers import sequence_mask_mxlen
from baseline.reader import register_reader, SeqLabelReader
import torch
from torch.utils.data import DataLoader
from audio8.wav2vec2 import Wav2Vec2Encoder, Wav2Vec2PooledEncoder
import numpy as np
import os
from eight_mile.pytorch.layers import EmbeddingsStack
from baseline.pytorch import TensorDef, BaseLayer
import time
import soundfile as sf
import csv


class CSVDictReader(SeqLabelReader):

    def get_label(self, entry):
        pass

    def get_label_index(self, entry):
        return self.label2index[self.get_label(entry)]

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__()
        self.datasets = {}
        self.labels = []
        self.label2index = {}
        self.dataset_dir = None

    def build_vocab(self, files, **kwargs):
        for f in files:
            self.read_csv(f)
        return {}, self.labels

    def read_csv(self, filename):
        if filename in self.datasets:
            return
        dataset = []
        labels = set(self.labels)
        self.dataset_dir = os.path.dirname(filename)
        with open(filename) as csvfile:
            d = csv.DictReader(csvfile)
            for entry in d:
                dataset.append(entry)
                labels.add(self.get_label(entry))
        self.datasets[filename] = dataset
        self.labels = list(labels)
        self.label2index = {l: i for i, l in enumerate(self.labels)}

    def collate(self, batch_list):
        pass

    def load(self, filename, vocabs, batchsz, **kwargs):
        self.read_csv(filename)
        shuffle = kwargs.get('shuffle', False)
        return DataLoader(self.datasets[filename], batch_size=batchsz, collate_fn=self.collate, shuffle=shuffle)


@register_reader(task='classify', name='fluent')
class FluentDataReader(CSVDictReader):

    def get_label(self, entry):
        label = '_'.join([entry['action'], entry['object'], entry['location']])
        return label

    def process_sample(self, file):
        """Read in a line and turn it into an entry.  FIXME, get from anywhere

        The entries will get collated by the data loader

        :param file:
        :return:
        """
        wav, _ = sf.read(file)
        wav = wav.astype(np.float32)
        return wav

    def collate(self, batch_list):
        audio = [self.process_sample(os.path.join(self.dataset_dir, '..', f['path'])) for f in batch_list]
        audio_lengths = torch.IntTensor([len(x) for x in audio])
        mxlen = audio_lengths.max()
        audio_padded = np.zeros((len(audio), mxlen), dtype=np.float32)

        for b in range(len(audio)):
            audio_padded[b, :len(audio[b])] = audio[b]

        actions = torch.tensor([self.get_label_index(f) for f in batch_list])
        return {'audio': (torch.from_numpy(audio_padded), sequence_mask_mxlen(audio_lengths, mxlen),), 'y': actions}


class Wav2Vec2PooledEmbeddings(PyTorchEmbeddings):

    def __init__(self, **kwargs):
        super().__init__()
        reduction_type = kwargs.get('reduction_type', 'max')
        self.d_model = int(kwargs.get('dsz', kwargs.get('d_model', 768)))
        self.encoder = Wav2Vec2PooledEncoder(d_model=self.d_model, reduction_type=reduction_type)
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
