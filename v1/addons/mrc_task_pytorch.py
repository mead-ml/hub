"""Utilities for running MRC

This includes some utilities for MRC problems, including an iterable PyTorch loader that doesnt require examples
to fit in core memory.

"""
import numpy as np
from eight_mile.utils import Offsets, Average, listify
from eight_mile.pytorch.layers import WithDropout, Dense
from baseline.utils import get_model_file, get_metric_cmp
from baseline.reader import register_reader
from baseline.model import register_model
import torch
import six
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import IterableDataset, TensorDataset
import collections
import json
import torch.nn as nn
import logging
from eight_mile.progress import create_progress_bar
import string
import re
import os
from eight_mile.pytorch.optz import OptimizerManager
from baseline.train import register_training_func, register_trainer, EpochReportingTrainer, create_trainer
from baseline.vectorizers import convert_tokens_to_ids
from baseline.model import create_model_for
from typing import List

from mead.tasks import Task, Backend, register_task
from mead.utils import read_config_file_or_json, index_by_label, print_dataset_info
from eight_mile.downloads import DataDownloader
import math
import regex
logger = logging.getLogger('baseline')

# Use for GPT2, RoBERTa, Longformer
BPE_PATTERN = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def bpe_tokenize(s, strip_ws=True):
    s_out = regex.findall(BPE_PATTERN, s)
    return s_out if not strip_ws else [w.strip() for w in s_out]

def bu_tokenize(s, strip_ws=True):
    import toky
    s_out = [s.get_text() for s in toky.bu_assembly(s)]
    return [s for s in s_out if s not in ['<', '>']]

def whitespace_tokenize(s, strip_ws=True):
    return s.split()

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

class MRCExample:
    """Intermediate object that holds a single QA sample, this gets converted to features later
    """
    def __init__(self,
                 qas_id,
                 query_item,
                 doc_tokens=None,
                 answers=None,
                 #orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.query_item = query_item
        self.doc_tokens = doc_tokens
        #self.orig_answer_text = orig_answer_text
        self.answers = answers
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = f"qas_id: {self.qas_id}, query_item: {self.query_item}\ncontext_item: {' '.join(self.doc_tokens)}"
        if self.start_position:
            s += f"\nstart_position: {self.start_position}"
        if self.start_position:
            s += f"\nend_position: {self.end_position}"
        if self.start_position:
            s += f"\nis_impossible: {self.is_impossible}"
        return s

class InputFeatures:

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        # self.span_position = span_position
        self.is_impossible = is_impossible


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def read_examples(input_file, is_training):
    """Read SQuaD style formatted examples, both v1.1 and v2

    For v1.1, the is_impossible field is absent, so default that to False here to support both.


    :param input_file:
    :param is_training:
    :return:
    """
    examples = []
    with open(input_file, "r") as f:
        input_data = json.load(f)['data']
        pg = create_progress_bar(len(input_data))
        for entry in pg(input_data):
            for paragraph in entry['paragraphs']:
                paragraph_text = paragraph['context']
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)
                for qa in paragraph['qas']:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position = None
                    end_position = None
                    #orig_answer_text = None
                    is_impossible = False
                    if is_training:

                        is_impossible = bool(qa.get('is_impossible', False))
                        # The dev set has more than one example possibly
                        # The BERT code raises an error, which makes sense when eval is offline
                        if not is_impossible:
                            all_answers = []
                            skip_example = False
                            for ai, answer in enumerate(qa['answers']):
                                # For training we have a single answer, for dev the scoring takes into  account all answers
                                # so in order to do this the way we want with inline eval we need to handle this, right now
                                # our scores are too low because we only use the first
                                if ai == 0:

                                    orig_answer_text = answer['text']
                                    answer_offset = answer['answer_start']
                                    answer_length = len(orig_answer_text)
                                    start_position = char_to_word_offset[answer_offset]
                                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                                    actual_text = ' '.join(
                                        doc_tokens[start_position:(end_position + 1)]

                                    )
                                    cleaned_answer_text = ' '.join(whitespace_tokenize(orig_answer_text))

                                    if actual_text.find(cleaned_answer_text) == -1:
                                        logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                                        skip_example = True
                                        break
                                if answer['text'] not in all_answers:
                                    all_answers.append(answer['text'])
                            if skip_example:
                                continue
                        # This should only happen outside of mead-train for offline evaluation
                        else:
                            start_position = -1
                            end_position = -1
                            all_answers = []

                    example = MRCExample(qas_id,
                                         question_text,
                                         doc_tokens,
                                         all_answers,
                                         start_position,
                                         end_position,
                                         is_impossible)
                    examples.append(example)
    return examples


class MRCDatasetIterator(IterableDataset):

    def __init__(self, input_file, vectorizer, mxlen=384, has_impossible=True, is_training=True, doc_stride=128, mxqlen=64,
                 shuffle=True, tok_type=None, strip_ws=True):
        super().__init__()
        self.vectorizer = vectorizer
        self.CLS_TOKEN = '[CLS]'
        if '<EOU>' in self.vectorizer.vocab:
            self.EOU_TOKEN = '<EOU>'
        elif '[SEP]' in self.vectorizer.vocab:
            self.EOU_TOKEN = '[SEP]'
        else:
            self.EOU_TOKEN = Offsets.VALUES[Offsets.EOS]
        print('SEP token', self.EOU_TOKEN)
        self.input_file = input_file
        self.mxlen = mxlen
        self.doc_stride = doc_stride
        self.mxqlen = mxqlen
        self.has_impossible = has_impossible
        self.is_training = is_training

        self.tokenizer_fn = whitespace_tokenize
        if tok_type == 'pretok' or tok_type == 'bpe':
            logger.warning("Doing GPT-style pre-tokenization. This may not be necessary for WordPiece vectorizers")
            self.tokenizer_fn = bu_tokenize
        elif tok_type == 'toky':
            logger.warning("Doing toky tokenization.")
            self.tokenizer_fn = bu_tokenize
        self.strip_ws = strip_ws
        if self.strip_ws:
            logger.warning("Stripping leading whitespace on tokens.  This may not be required for GPT*, RoBERTa or variants")
        self.examples = read_examples(input_file, is_training)
        # Add a tokenized version to all examples
        for example in self.examples:
            example.answers = [' '.join(self.tokenize(a)) for a in example.answers]
        self.shuffle = shuffle

    def _improve_answer_span(self, doc_tokens, input_start, input_end, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        # The SQuAD annotations are character based. We first project them to
        # whitespace-tokenized words. But then after WordPiece tokenization, we can
        # often find a "better match". For example:
        #
        #   Question: What year was John Smith born?
        #   Context: The leader was John Smith (1895-1943).
        #   Answer: 1895
        #
        # The original whitespace-tokenized answer will be "(1895-1943).". However
        # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
        # the exact answer, 1895.
        #
        # However, this is not always possible. Consider the following:
        #
        #   Question: What country is the top exporter of electornics?
        #   Context: The Japanese electronics industry is the lagest in the world.
        #   Answer: Japan
        #
        # In this case, the annotator chose "Japan" as a character sub-span of
        # the word "Japanese". Since our WordPiece tokenizer does not split
        # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
        # in SQuAD, but does happen.

        # TODO: For right now we dont need this because we will always be tokenizing our orig_answer_text
        ##tok_answer_text = " ".join(self.tokenize(orig_answer_text))
        tok_answer_text = orig_answer_text

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return new_start, new_end

        return input_start, input_end

    @property
    def num_examples(self):
        """Note that the number of examples is *not* the same as the number of samples for training

        The examples can be any length, the samples are typically 384 with a stride of 128, so if you have long
        examples they will almost definitely be cut up into multiple samples.


        :return:
        """
        return len(self.examples)

    def tokenize(self, text):
        """Encapsulate vectorizer tokenization interface

        :param text:
        :return:
        """
        return list(self.vectorizer.iterable(self.tokenizer_fn(text, self.strip_ws)))

    def convert_tokens_to_ids(self, tokens):
        """Encapsulate vectorizer interface

        :param tokens:
        :return:
        """
        return convert_tokens_to_ids(self.vectorizer.vocab, tokens)

    def __iter__(self):
        unique_id = 0
        # The len(self) is not correct if used in context of features, but it is correct WRT number of docs (examples)
        # Using it in this manner below to shuffle the examples is proper usage
        order = np.arange(self.num_examples)
        if self.shuffle:
            order = np.random.permutation(order)
        # [3, 6, 22, 8, 9]
        for example_index in order:
            example = self.examples[example_index]
            query_tokens = self.tokenize(example.query_item)

            if len(query_tokens) > self.mxqlen:
                query_tokens = query_tokens[0:self.mxqlen]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for i, token in enumerate(example.doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = self.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)

            tok_start_position = None
            tok_end_position = None
            if self.is_training and example.is_impossible:
                tok_start_position = -1
                tok_end_position = -1
            if self.is_training and not example.is_impossible:
                tok_start_position = orig_to_tok_index[example.start_position]
                if example.end_position < len(example.doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
                else:
                    tok_end_position = len(all_doc_tokens) - 1
                (tok_start_position, tok_end_position) = self._improve_answer_span(
                    all_doc_tokens, tok_start_position, tok_end_position,
                    example.answers[0])

            # The -3 accounts for [CLS], [SEP] and [SEP]
            max_tokens_for_doc = self.mxlen - len(query_tokens) - 3

            # We can have documents that are longer than the maximum sequence length.
            # To deal with this we do a sliding window approach, where we take chunks
            # of the up to our max length with a stride of `doc_stride`.
            _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                "DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, self.doc_stride)

            for doc_span_index, doc_span in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                # FIXME! Dont use a literal here, read from config
                tokens.append(self.CLS_TOKEN)
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                # FIXME! Dont use a literal here, read from config
                tokens.append(self.EOU_TOKEN)
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append(self.EOU_TOKEN)
                segment_ids.append(1)

                input_ids = self.convert_tokens_to_ids(tokens)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                #input_mask = [1] * len(input_ids)

                # Zero-pad up to the sequence length.
                while len(input_ids) < self.mxlen:
                    input_ids.append(0)
                    #input_mask.append(0)
                    segment_ids.append(0)

                start_position = None
                end_position = None
                if self.is_training and not example.is_impossible:
                    # For training, if our document chunk does not contain an annotation
                    # we throw it out, since there is nothing to predict.
                    doc_start = doc_span.start
                    doc_end = doc_span.start + doc_span.length - 1
                    out_of_span = False
                    if not (tok_start_position >= doc_start and
                            tok_end_position <= doc_end):
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

                if self.is_training and example.is_impossible:
                    start_position = 0
                    end_position = 0

                # Because we will form this on an epoch, the id of the feature according its epoch ordering is
                # sufficient as a unique id.
                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    # We do not need this as our embeddings compute it on the fly
                    #input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)

                # Run callback
                unique_id += 1
                yield feature


def create_collate_on(field='word'):
    """Closure for creating a collate function for use with the DataLoader using some feature name

    :param field:
    :return:
    """
    def collate_fn(batch_list):
        example_index = torch.tensor([f.example_index for f in batch_list])
        input_ids = torch.tensor([f.input_ids for f in batch_list])
        unique_id = torch.tensor([f.unique_id for f in batch_list])
        segment_ids = torch.tensor([f.segment_ids for f in batch_list])
        start_pos = torch.tensor([f.start_position for f in batch_list])
        end_pos = torch.tensor([f.end_position for f in batch_list])
        token_is_max_context = [f.token_is_max_context for f in batch_list]
        token_to_orig_map = [f.token_to_orig_map for f in batch_list]
        tokens = [f.tokens for f in batch_list]
        return {field: input_ids, 'token_type': segment_ids,
                'unique_id': unique_id,
                'token_is_max_context': token_is_max_context,
                'token_to_orig_map': token_to_orig_map,
                'example_index': example_index, 'start_pos': start_pos, 'end_pos': end_pos, 'tokens': tokens}
    return collate_fn


@register_reader(task='mrc', name='default')
class SQuADJsonReader:

    def load(self, filename, index, batchsz, **kwargs):
        pass

    def __init__(self, vectorizers, trim=False, truncate=False, **kwargs):
        super().__init__()

        self.label2index = {}
        self.vectorizers = vectorizers
        self.clean_fn = kwargs.get('clean_fn')
        if self.clean_fn is None:
            self.clean_fn = lambda x: x
        self.trim = trim
        self.truncate = truncate

    @property
    def vectorizer(self):
        return list(self.vectorizers.values())[0]

    def build_vocab(self, _, **kwargs):
        """Take a directory (as a string), or an array of files and build a vocabulary

        Take in a directory or an array of individual files (as a list).  If the argument is
        a string, it may be a directory, in which case, all files in the directory will be loaded
        to form a vocabulary.

        :param files: Either a directory (str), or an array of individual files
        :return:
        """
        key = list(self.vectorizers.keys())[0]
        return {key: self.vectorizers[key]}, self.get_labels()

    def get_labels(self):
        labels = [''] * len(self.label2index)
        for label, index in self.label2index.items():
            labels[index] = label
        return labels

    def load(self, filename, _, batchsz, **kwargs):
        """The load function normally takes in a vocab for the 3rd argument, but we are always going to assume predefined vocabs

        :param filename:
        :param _:
        :param batchsz:
        :param kwargs:
        :return:
        """
        shuffle = kwargs.get('shuffle', False)
        doc_stride = int(kwargs.get('doc_stride', 128))
        mxqlen = int(kwargs.get('mxqlen', 64))
        has_impossible = bool(kwargs.get('has_impossible', True))
        # Whenever we run in mead-train we want this to be on, because, unlike the BERT code, we do inline evaluation
        # and so we need ground truth
        is_training = kwargs.get('is_training', True)
        tok_type = kwargs.get('tok_type')
        strip_ws = kwargs.get('strip_ws', True)

        ds = MRCDatasetIterator(filename, self.vectorizer, mxlen=self.vectorizer.mxlen, doc_stride=doc_stride,
                                has_impossible=has_impossible, is_training=is_training, mxqlen=mxqlen, shuffle=shuffle,
                                tok_type=tok_type, strip_ws=strip_ws)

        dl = DataLoader(ds, batch_size=batchsz, pin_memory=False, collate_fn=create_collate_on())
        return dl
        #return Batcher(loader, batchsz)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def clean_answer(answer):
        # De-tokenize WordPieces that have been split off. #TODO: Add support for BPE
        return answer.replace(" ##", "").replace("##", "").replace('@@ ', '')

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    s = clean_answer(s)
    return ' '.join((remove_articles(remove_punc(s.lower()))).split())


def get_tokens(s: str) -> List[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        same = float(gold_toks == pred_toks)
        return same, same, same
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def compute_metrics(real_answers, pred_answer):
    exact_match = max([compute_exact(real_answer, pred_answer) for real_answer in real_answers])
    nearest_answer = real_answers[0]
    p_max, r_max, f1_max = compute_f1(real_answers[0], pred_answer)

    for real_answer in real_answers[1:]:
        p, r, f1 = compute_f1(real_answer, pred_answer)
        if f1 > f1_max:
            f1_max = f1
            p_max = p
            r_max = r
            nearest_answer = real_answer
    return exact_match, p_max, r_max, f1_max, nearest_answer


@register_trainer(task='mrc', name='default')
class MRCTrainerPyTorch(EpochReportingTrainer):

    def __init__(self, model, **kwargs):

        super().__init__()
        if type(model) is dict:
            model = create_model_for('mrc', **model)
        self.clip = float(kwargs.get('clip', 5))
        self.labels = model.labels
        self.gpus = int(kwargs.get('gpus', 1))
        if self.gpus == -1:
            self.gpus = len(os.getenv('CUDA_VISIBLE_DEVICES', os.getenv('NV_GPU', '0')).split(','))

        self.optimizer = OptimizerManager(model, **kwargs)
        logger.info("Model has {:,} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        self.model = model
        if self.gpus > 0 and self.model.gpu:
            self.crit = model.create_loss().cuda()
            if self.gpus > 1:
                self.model = torch.nn.DataParallel(model).cuda()
            else:
                self.model.cuda()
        else:
            logger.warning("Requested training on CPU.  This will be slow.")
            self.crit = model.create_loss()
            self.model = model
        self.nsteps = kwargs.get('nsteps', six.MAXSIZE)

    def _get_pytorch_model(self):
        return self.model.module if self.gpus > 1 else self.model

    def save(self, model_file):
        self._get_pytorch_model().save(model_file)

    def _make_input(self, batch_dict, **kwargs):
        return self._get_pytorch_model().make_input(batch_dict, **kwargs)

    @staticmethod
    def _get_batchsz(batch_dict):
        return len(batch_dict['tokens'])

    @staticmethod
    def _get_best_indices(logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def convert_prelim_preds_to_nbest(self, prelim_predictions, features_for_example, null_start, null_end, limit=20):
        """The original predictions per example are sorted and passed in with scores, but need to be N-bests

        BERT sample code both in official and HF repository allows adding multiple non-predictions, which can cause the
        entire N-best list to not contain a span answer.  In the BERT code, they add a `nonce` example with some random
        text in it, but it seems more sensible to take the highest ranking span answer and always add that instead.

        We also keep track of the gold answer here for later use and return that

        :param prelim_predictions: A list of preliminary predictions
        :param features_for_example: An index of features for this single example
        :param null_start: The null_start logit value
        :param null_end: Null end logit value
        :param limit: The N in nbest
        :return: A list of N-bests {'predict_text:[], 'start_logit':x, 'end_logit':y}
        """
        seen_answers = set()
        nbest = []
        for prediction in prelim_predictions:
            feature = features_for_example[prediction['feature_index']]
            # gold_start = feature['gold_start']
            # gold_end = feature['gold_end']
            # A couple of possibilities
            # The gold text might not be reachable in the answer, in which case the sample
            # will be showing the zero sample
            #
            # The gold start might be set already in which case we dont replace
            # We dont want to find out that the answer changed though, that should be an error

            # If the its pointing at token 0, thats not an answer, since that contains a special token

            # This test makes sure that we have hit the limit of nbest and that there are at least 2 types of answers
            # Usually this is '' and some other answer
            if len(nbest) == limit and len(seen_answers) > 1:
                break
            if prediction['start_index'] > 0:
                feature_text_tok = feature['tokens'][prediction['start_index']:prediction['end_index'] + 1]
                feature_text_str = ' '.join(feature_text_tok)
                if feature_text_str not in seen_answers:
                    nbest.append({'predict_text': feature_text_tok,
                                  'start_logit': prediction['start_logit'],
                                  'end_logit': prediction['end_logit']})
                    seen_answers.add(feature_text_str)
            else:
                seen_answers.add('')
                if len(nbest) < limit:
                    nbest.append(
                        {'predict_text': [], 'start_logit': null_start, 'end_logit': null_end})
        if '' not in seen_answers:
            nbest.append({'predict_text': [], 'start_logit': null_start, 'end_logit': null_end})
        # This shouldnt happen anymore because we fixed the BERT logic above by adding the highest non-null score
        #if len(seen_answers) < 2 and '' in seen_answers:
        #    raise Exception("We dont have any non-null guesses!")

        return nbest

    def create_prelim_predictions(self, features_for_example, limit_nbest, limit_answer_length):
        prelim_predictions = []
        score_null = 1e8
        null_start = 0
        min_null_feature_index = 0
        null_end = 0
        ## Each softmax is the probability of each token, so when we sort it and maintain the indices we get what we need
        for fi, feature in enumerate(features_for_example):
            start_indices = self._get_best_indices(feature['start'], limit_nbest)
            end_indices = self._get_best_indices(feature['end'], limit_nbest)
            feature_null_score = feature['start'][0] + feature['end'][0]
            if feature_null_score < score_null:
                score_null = feature_null_score
                null_start = feature['start'][0]
                null_end = feature['end'][0]
                min_null_feature_index = fi
            for start_index in start_indices:
                for end_index in end_indices:
                    if start_index >= len(feature['tokens']):
                        continue
                    if end_index >= len(feature['tokens']):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > limit_answer_length:
                        continue
                    if start_index not in feature['token_to_orig_map']:
                    #    logger.warning("Start is not in orig mapping")
                        continue
                    if end_index not in feature['token_to_orig_map']:
                    #    logger.warning("End is not in orig mapping")
                        continue
                    if not feature['token_is_max_context'].get(start_index, False):
                    #    logger.warning("Token isnt max context. Skipping")
                        continue
                    prelim_predictions.append({
                        'feature_index': fi,
                        'start_index': start_index,
                        'end_index': end_index,
                        'start_logit': feature['start'][start_index],
                        'end_logit': feature['end'][end_index]})

            prelim_predictions.append({
                'feature_index': min_null_feature_index,
                'start_index': 0,
                'end_index': 0,
                'start_logit': null_start,
                'end_logit': null_end})

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x['start_logit'] + x['end_logit']),
            reverse=True)
        return prelim_predictions, null_start, null_end, min_null_feature_index, score_null

    def get_final_text(self, vectorizer, pred_text, orig_text):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heruistic between
        # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.

        tok_text = " ".join(vectorizer.iterable(orig_text.split()))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            logger.warning(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            logger.warning("Length not equal after stripping spaces: '%s' vs '%s'",
                           orig_ns_text, tok_ns_text)
        return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            logger.warning("Couldn't map start position")
        return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            logger.warning("Couldn't map end position")
        return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text

    def _test(self, loader, **kwargs):

        self.model.eval()

        null_score_diff_threshes = listify(kwargs.get('null_score_diff_threshes', np.arange(-5, 1).tolist()))

        limit_nbest = kwargs.get('limit_nbest', 20)
        limit_answer_length = kwargs.get('limit_answer_length', 30)
        exact_matches = {thresh: Average(f'exact_match@{thresh}') for thresh in null_score_diff_threshes}
        precisions = {thresh: Average(f'precision@{thresh}') for thresh in null_score_diff_threshes}
        recalls = {thresh: Average(f'recall@{thresh}') for thresh in null_score_diff_threshes}
        f1s = {thresh: Average(f'f1@{thresh}') for thresh in null_score_diff_threshes}
        example_index_to_features, total_loss, total_norm = self.eval_model(loader)
        # For each example, we need to go through and update info

        all_ids = list(example_index_to_features.keys())
        example_index_to_predictions = {}
        # we are going to go over all the examples, convert them to predictions
        # convert those predictions to nbests
        for i, id in enumerate(all_ids):

            features_for_example = example_index_to_features[id]
            prelim_predictions, null_start, null_end, min_null_feature_index, score_null = self.create_prelim_predictions(features_for_example, limit_nbest, limit_answer_length)
            nbest = self.convert_prelim_preds_to_nbest(prelim_predictions, features_for_example, null_start, null_end, limit_nbest)

            # This is the BERT routine for figuring out the null values, if you have a different model this might change
            total_scores = []
            best_non_null_entry = None
            for entry in nbest:
                total_scores.append(entry['start_logit'] + entry['end_logit'])
                if not best_non_null_entry:
                    if entry['predict_text']:
                        best_non_null_entry = entry

            probs = _compute_softmax(total_scores)
            for entry in nbest:
                entry['probs'] = probs

            example_index_to_predictions[id] = nbest
            if best_non_null_entry:
                score_diff = score_null - best_non_null_entry['start_logit'] - best_non_null_entry['end_logit']
            else:
                score_diff = score_null
            # The BERT codebase suggests tuning the threshold here between -1 and -5 based on the dev data.
            # Here we will run it over all integers in that range to find a suitable value
            for null_score_diff_thresh in null_score_diff_threshes:
                # This answer is not suitable for SQuaD evalution, because it only contains the tokenized data
                # There is a more complex way of handling the final results when creating a SQuaD result since it
                # needs to be able to get back to the SQuaD tokenized answer.  However, for just calculating the
                # metrics while doing evaluation, we dont really care about this.  This could emit a metric score that
                # is slightly different from the score we would get if submitting to SQuaD but it should be close enough
                # for things like early stopping
                if score_diff > null_score_diff_thresh:
                    pred_answer = ''
                else:
                    pred_answer = ' '.join(best_non_null_entry['predict_text'])

                # We can look up the original item to find the gold answers
                example = loader.dataset.examples[id]
                real_answers = example.answers
                if example.is_impossible:
                    real_answers = ['']
                exact_match, precision, recall, f1, nearest_answer = compute_metrics(real_answers, pred_answer)
                exact_matches[null_score_diff_thresh].update(float(exact_match))
                precisions[null_score_diff_thresh].update(precision)
                recalls[null_score_diff_thresh].update(recall)
                f1s[null_score_diff_thresh].update(f1)

        metrics = {}
        metrics['avg_loss'] = total_loss / float(total_norm)

        metrics.update({m.name: m.avg for m in precisions.values()})
        metrics.update({m.name: m.avg for m in recalls.values()})
        metrics.update({m.name: m.avg for m in f1s.values()})
        metrics.update({m.name: m.avg for m in exact_matches.values()})

        # Shouldnt really do early stopping I guess since it could be related to thresholding, but allow for now
        metrics['precision'] = max(m.avg for m in precisions.values())
        metrics['recall'] = max(m.avg for m in recalls.values())
        metrics['f1'] = max(m.avg for m in f1s.values())
        metrics['exact_match'] = max(m.avg for m in exact_matches.values())

        return metrics

    def eval_model(self, loader):
        """Run evaluation over and entire epoch of data

        :param loader: The data loader
        :param pg: The progress bar wrapper
        :return:
        """
        total_loss = 0
        total_norm = 0
        example_index_to_features = collections.defaultdict(list)
        for batch_dict in loader:
            with torch.no_grad():
                example = self._make_input(batch_dict)
                y_start_pos = example.pop('start_pos')
                y_end_pos = example.pop('end_pos')
                start_pos_pred, end_pos_pred = self.model(example)
                loss = self.crit(start_pos_pred, end_pos_pred, y_start_pos, y_end_pos)
                batchsz = self._get_batchsz(example)
                total_loss += loss.item() * batchsz
                total_norm += batchsz
                start_pos_pred = start_pos_pred.detach().cpu()
                end_pos_pred = end_pos_pred.detach().cpu()
                y_start_pos = y_start_pos.detach().cpu()
                y_end_pos = y_end_pos.detach().cpu()

                for i in range(batchsz):
                    tokens = example['tokens'][i]
                    unique_id = example['unique_id'][i].item()
                    example_id = example['example_index'][i].item()
                    token_to_orig_map = example['token_to_orig_map'][i]
                    token_is_max_context = example['token_is_max_context'][i]

                    # Gather all the results
                    example_index_to_features[example_id].append({'unique_id': unique_id,
                                                                  'tokens': tokens,
                                                                  'start': start_pos_pred[i].tolist(),
                                                                  'end': end_pos_pred[i].tolist(),
                                                                  'gold_start': y_start_pos[i].tolist(),
                                                                  'gold_end': y_end_pos[i].tolist(),
                                                                  'token_to_orig_map': token_to_orig_map,
                                                                  'token_is_max_context': token_is_max_context
                                                                  })
        return example_index_to_features, total_loss, total_norm

    def _make_input(self, batch_dict):
        ex = {}
        for k, v in batch_dict.items():
            # TODO: clean this up!
            ex[k] = v.cuda() if k not in ['tokens', 'unique_id', 'example_index', 'token_to_orig_map', 'token_is_max_context'] else v
        return ex

    def _train(self, loader, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        limit_samples = kwargs.get('limit_samples', None)
        epoch_loss = 0
        epoch_div = 0
        for i, batch_dict in enumerate(loader):
            if limit_samples and i > limit_samples:
                break
            self.optimizer.zero_grad()
            example = self._make_input(batch_dict)
            y_start_pos = example.pop('start_pos')
            y_end_pos = example.pop('end_pos')
            batchsz = self._get_batchsz(example)
            #print(y_start_pos, y_end_pos)
            start_pos_pred, end_pos_pred = self.model(example)
            loss = self.crit(start_pos_pred, end_pos_pred, y_start_pos, y_end_pos)
            report_loss = loss.item() * batchsz
            epoch_loss += report_loss
            epoch_div += batchsz
            self.nstep_agg += report_loss
            self.nstep_div += batchsz
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            if (self.optimizer.global_step + 1) % self.nsteps == 0:
                metrics = self.calc_metrics(self.nstep_agg, self.nstep_div)
                metrics['lr'] = self.optimizer.current_lr
                self.report(
                    self.optimizer.global_step + 1, metrics, self.nstep_start,
                    'Train', 'STEP', reporting_fns, self.nsteps
                )
                self.reset_nstep()
        metrics['lr'] = self.optimizer.current_lr
        metrics['avg_loss'] = epoch_loss / float(epoch_div)
        return metrics


@register_training_func('mrc')
def fit(model_params, ts, vs, es, **kwargs):
    """
    Train a classifier using PyTorch
    :param model_params: The model to train
    :param ts: A training data set
    :param vs: A validation data set
    :param es: A test data set, can be None
    :param kwargs: See below

    :Keyword Arguments:
        * *do_early_stopping* (``bool``) -- Stop after eval data is not improving. Default to True
        * *epochs* (``int``) -- how many epochs.  Default to 20
        * *outfile* -- Model output file, defaults to classifier-model.pyth
        * *patience* --
           How many epochs where evaluation is no longer improving before we give up
        * *reporting* --
           Callbacks which may be used on reporting updates
        * *optim* --
           Optimizer to use, defaults to `sgd`
        * *eta, lr* (``float``) --
           Learning rate, defaults to 0.01
        * *mom* (``float``) --
           Momentum (SGD only), defaults to 0.9 if optim is `sgd`
    :return:
    """
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = kwargs.get('verbose', {'console': kwargs.get('verbose_console', False), 'file': kwargs.get('verbose_file', None)})
    epochs = int(kwargs.get('epochs', 20))
    model_file = get_model_file('mrc', 'pytorch', kwargs.get('basedir'))
    output = kwargs.get('output')
    txts = kwargs.get('txts')
    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'f1')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.pop('reporting', []))
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns, **kwargs)
        test_metrics = trainer.test(vs, reporting_fns, **kwargs)

        if do_early_stopping is False:
            trainer.save(model_file)

        elif early_stopping_cmp(test_metrics[early_stopping_metric], best_metric):
            last_improved = epoch
            best_metric = test_metrics[early_stopping_metric]
            logger.info('New best %.3f', best_metric)
            trainer.save(model_file)

        elif (epoch - last_improved) > patience:
            logger.info('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        logger.info('Best performance on %s: %.3f at epoch %d', early_stopping_metric, best_metric, last_improved)

    if es is not None:
        logger.info('Reloading best checkpoint')
        model = torch.load(model_file)
        trainer = create_trainer(model, **kwargs)
        test_metrics = trainer.test(es, reporting_fns, phase='Test', verbose=verbose, output=output, txts=txts)
    return test_metrics


@register_task
class MRCTask(Task):

    def __init__(self, mead_settings_config, **kwargs):
        super().__init__(mead_settings_config, **kwargs)

    @classmethod
    def task_name(cls):
        return 'mrc'

    def _create_backend(self, **kwargs):
        backend = Backend(self.config_params.get('backend', 'tf'), kwargs)
        backend.load(self.task_name())

        return backend

    def _setup_task(self, **kwargs):
        super()._setup_task(**kwargs)
        self.config_params.setdefault('preproc', {})
        self.config_params['preproc']['clean_fn'] = None

    def initialize(self, embeddings):
        embeddings = read_config_file_or_json(embeddings, 'embeddings')
        embeddings_set = index_by_label(embeddings)
        self.dataset = DataDownloader(self.dataset, self.data_download_cache).download()
        print_dataset_info(self.dataset)

        vocab_sources = [self.dataset['train_file'], self.dataset['valid_file']]
        # TODO: make this optional
        if 'test_file' in self.dataset:
            vocab_sources.append(self.dataset['test_file'])

        vocab, self.labels = self.reader.build_vocab(vocab_sources,
                                                     min_f=Task._get_min_f(self.config_params),
                                                     vocab_file=self.dataset.get('vocab_file'),
                                                     label_file=self.dataset.get('label_file'))
        self.embeddings, self.feat2index = self._create_embeddings(embeddings_set, vocab, self.config_params['features'])
        #baseline.save_vocabs(self.get_basedir(), self.feat2index)

    def _get_features(self):
        return self.embeddings

    def _get_labels(self):
        return self.labels

    def _reorganize_params(self):
        train_params = self.config_params['train']
        train_params['batchsz'] = train_params['batchsz'] if 'batchsz' in train_params else self.config_params['batchsz']
        train_params['test_batchsz'] = train_params.get('test_batchsz', self.config_params.get('test_batchsz', 1))
        unif = self.config_params.get('unif', 0.1)
        model = self.config_params['model']
        model['unif'] = model.get('unif', unif)
        lengths_key = model.get('lengths_key', self.primary_key)
        if lengths_key is not None:
            if not lengths_key.endswith('_lengths'):
                lengths_key = '{}_lengths'.format(lengths_key)
            model['lengths_key'] = lengths_key
        if self.backend.params is not None:
            for k, v in self.backend.params.items():
                model[k] = v

    def _load_dataset(self):
        read = self.config_params['reader'] if 'reader' in self.config_params else self.config_params['loader']
        sort_key = read.get('sort_key')
        bsz, vbsz, tbsz = Task._get_batchsz(self.config_params)
        self.train_data = self.reader.load(
            self.dataset['train_file'],
            self.feat2index,
            bsz,
            shuffle=True,
            sort_key=sort_key,
            **read,
        )
        self.valid_data = self.reader.load(
            self.dataset['valid_file'],
            self.feat2index,
            vbsz,
            **read
        )
        self.test_data = None
        if 'test_file' in self.dataset:
            self.test_data = self.reader.load(
                self.dataset['test_file'],
                self.feat2index,
                tbsz,
                **read
            )

class StartAndEndLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_start = kwargs.get('weight_start', 0.5)
        self.weight_end = kwargs.get('weight_end', 0.5)
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, start_logits, end_logits, truth_start, truth_end):
        start_loss = self.loss_fct(start_logits, truth_start)
        end_loss = self.loss_fct(end_logits, truth_end)
        total_loss = self.weight_start * start_loss + self.weight_end * end_loss
        return total_loss


@register_model('mrc', 'default')
class BertQueryMRC(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def create_layers(self, embeddings, **kwargs):
        self.embeddings = self.init_embed(embeddings)
        dropout = kwargs.get('pdrop', 0.1)
        hidden_dims = listify(kwargs.get('hsz', []))
        if not hidden_dims:
            last_hsz = self.embeddings.get_dsz()
            self.proj = nn.Identity()
        else:
            current_hsz = self.embeddings.get_dsz()
            last_hsz = hidden_dims[-1]
            self.proj = nn.Sequential([WithDropout(Dense(current_hsz, hsz, activation='relu'), dropout) for hsz in hidden_dims])

        self.start_end_outputs = nn.Linear(last_hsz, 2)


    def save(self, outname: str):
        """Save out the model

        :param outname: The name of the checkpoint to write
        :return:
        """
        torch.save(self, outname)

    @classmethod
    def create(cls, embeddings, labels, **kwargs) -> 'BertQueryMRC':
        """Create a span finder from the inputs.  Most classes shouldnt extend this

        :param embeddings: A dictionary containing the input feature indices
        :param labels: A list of the labels (tags)
        :param kwargs: See below

        :Keyword Arguments:

        * *lengths_key* (`str`) Which feature identifies the length of the sequence
        * *activation* (`str`) What type of activation function to use (defaults to `tanh`)
        * *dropout* (`str`) What fraction dropout to apply
        * *dropin* (`str`) A dictionarwith feature keys telling what fraction of word masking to apply to each feature

        :return:
        """
        model = cls()
        model.feature_key = kwargs.get('lengths_key', 'word_lengths').replace('_lengths', '')
        model.pdrop = float(kwargs.get('dropout', 0.5))
        model.dropin_values = kwargs.get('dropin', {})
        model.labels = labels
        model.gpu = not bool(kwargs.get('nogpu', False))
        model.create_layers(embeddings, **kwargs)
        return model

    def init_embed(self, embeddings, **kwargs):
        """This method creates the "embedding" layer of the inputs, with an optional reduction

        :param embeddings: A dictionary of embeddings

        :Keyword Arguments: See below
        * *embeddings_reduction* (defaults to `concat`) An operator to perform on a stack of embeddings

        :return: The output of the embedding stack followed by its reduction.  This will typically be an output
          with an additional dimension which is the hidden representation of the input
        """
        return embeddings[self.feature_key]

    def forward(self, features):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]]
            span_positions: (batch x max_len x max_len)
                span_positions[k][i][j] is one of [0, 1],
                span_positions[k][i][j] represents whether or not from start_pos{i} to end_pos{j} of the K-th sentence in the batch is an entity.
        """
        embedded = self.embeddings(features[self.feature_key], features['token_type'])
        embedded = self.proj(embedded)
        logits = self.start_end_outputs(embedded)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits

    def create_loss(self, **kwargs):
        return StartAndEndLoss(**kwargs)


# Num training samples: 132591, ~70s to load
if __name__ == '__main__':
    import time
    from baseline.vectorizers import WordpieceVectorizer1D
    input_file = '/data/datasets/mrc/train-v2.0.json'
    vectorizer = WordpieceVectorizer1D(vocab_file='https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt', mxlen=384)
    l = MRCDatasetIterator(input_file, vectorizer, mxlen=384)
    start = time.time()
    for i, x in enumerate(l):
        if i % 100000 == 0:
            print(i)
    print(i)
    elapsed = time.time() - start
    print(f'{elapsed} sec')