from eight_mile.pytorch.layers import sequence_mask_mxlen
from baseline.reader import register_reader, SeqLabelReader
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import soundfile as sf
import csv


class CSVDictReader(SeqLabelReader):

    def get_label(self, entry):
        pass

    def get_label_index(self, entry):
        return self.label2index[self.get_label(entry)]

    def __init__(self, vectorizers=None, trim=False, truncate=False, **kwargs):
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

    def load(self, filename, vocabs={}, batchsz=32, shuffle=False, **kwargs):
        self.read_csv(filename)
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
        transcripts = [f['transcription'] for f in batch_list]
        audio_lengths = torch.IntTensor([len(x) for x in audio])
        mxlen = audio_lengths.max()
        audio_padded = np.zeros((len(audio), mxlen), dtype=np.float32)

        for b in range(len(audio)):
            audio_padded[b, :len(audio[b])] = audio[b]

        actions = torch.tensor([self.get_label_index(f) for f in batch_list])
        return {'audio': (torch.from_numpy(audio_padded), sequence_mask_mxlen(audio_lengths, mxlen),),
                'transcript': transcripts,
                'y': actions}


