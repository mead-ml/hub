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
logger = logging.getLogger('baseline')


def whitespace_tokenize(s):
    return s.split()


class MRCExample:
    """Intermediate object that holds a single QA sample, this gets converted to features later
    """
    def __init__(self,
                 qas_id,
                 query_item,
                 doc_tokens=None,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.query_item = query_item
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
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
    """Features for a sample

    Because we are using our own Transformers, we dont have to pass in the valid mask, that gets computed internally.

    :param start_pos: start position is a list of symbol
    :param end_pos: end position is a list of symbol

    """

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
    examples = []
    with open(input_file, "r") as f:
        input_data = json.load(f)['data']
        pg = create_progress_bar(len(input_data))
        for entry in pg(input_data):
            for paragraph in entry['paragraphs']:
                context_item = paragraph['context']
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in context_item:
                    if MRCDatasetIterator.is_whitespace(c):
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
                    query_item = qa["question"]
                    start_position = None
                    end_position = None
                    orig_answer_text = None
                    is_impossible = False
                    if is_training:

                        is_impossible = bool(qa.get('is_impossible', False))
                        ## Need a solution here
                        ##if (len(qa["answers"]) != 1) and (not is_impossible):
                        ##    raise ValueError(
                        ##        "For training, each question should have exactly 1 answer."
                        ##    )
                        if not is_impossible:
                            answer = qa['answers'][0]
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
                                continue
                        else:
                            start_position = -1
                            end_position = -1
                            orig_answer_text = ''

                    example = MRCExample(qas_id,
                                         query_item,
                                         doc_tokens,
                                         orig_answer_text,
                                         start_position,
                                         end_position,
                                         is_impossible)
                    examples.append(example)
    return examples


class MRCDatasetIterator(IterableDataset):

    def __init__(self, input_file, vectorizer, mxlen=384, has_impossible=True, is_training=True, doc_stride=128, mxqlen=64):
        super().__init__()
        self.vectorizer = vectorizer
        self.input_file = input_file
        self.mxlen = mxlen
        self.doc_stride = doc_stride
        self.mxqlen = mxqlen
        self.has_impossible = has_impossible
        self.is_training = is_training
        self.examples = read_examples(input_file, is_training)

    def _improve_answer_span(self, doc_tokens, input_start, input_end, orig_answer_text):
        """Returns tokenized answer spans that better match the annotated answer."""
        tokenizer = self.vectorizer.tokenizer
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
        #tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))
        tok_answer_text = " ".join(list(self.vectorizer.iterable(whitespace_tokenize(orig_answer_text))))

        for new_start in range(input_start, input_end + 1):
            for new_end in range(input_end, new_start - 1, -1):
                text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
                if text_span == tok_answer_text:
                    return new_start, new_end

        return input_start, input_end

    def __len__(self):
        return len(self.examples)

    @staticmethod
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def tokenize(self, text):
        #return self.vectorizer.tokenizer.tokenize(text)
        return list(self.vectorizer.iterable(whitespace_tokenize(text)))

    def convert_tokens_to_ids(self, tokens):
        return convert_tokens_to_ids(self.vectorizer.vocab, tokens)
        #return self.vectorizer.tokenizer.convert_tokens_to_ids(tokens)

    def __iter__(self):

        unique_id = 1000000000
        shuffle = np.random.permutation(np.arange(len(self)))

        for si in shuffle:
            example_index = shuffle[si]

        #for (example_index, example) in enumerate(self.examples):
            example = self.examples[example_index]
            query_tokens = self.tokenize(example.query_item)

            if len(query_tokens) > self.mxqlen:
                query_tokens = query_tokens[0:self.mxqlen]

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(example.doc_tokens):
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
                    example.orig_answer_text)

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

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                tokens = []
                token_to_orig_map = {}
                token_is_max_context = {}
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in query_tokens:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append("[SEP]")
                segment_ids.append(0)

                for i in range(doc_span.length):
                    split_token_index = doc_span.start + i
                    token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                    is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                           split_token_index)
                    token_is_max_context[len(tokens)] = is_max_context
                    tokens.append(all_doc_tokens[split_token_index])
                    segment_ids.append(1)
                tokens.append("[SEP]")
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

                if example.is_impossible:
                    start_position = 0
                    end_position = 0

                feature = InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    #input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible)

                # Run callback
                yield feature
                unique_id += 1


class Batcher(IterableDataset):

    def __init__(self, dataset, batchsz, feature_key='word'):
        self.batchsz = batchsz
        self.dataset = dataset
        self.feature_key = feature_key

    def __len__(self):
        return len(self.dataset)//self.batchsz

    def _batch(self, batch_list):
        input_ids = torch.tensor([f.input_ids for f in batch_list])
        segment_ids = torch.tensor([f.segment_ids for f in batch_list])
        start_pos = torch.tensor([f.start_position for f in batch_list])
        end_pos = torch.tensor([f.end_position for f in batch_list])
        tokens = []
        for f in batch_list:
            tokens.append(f.tokens)
        return {self.feature_key: input_ids, 'token_type': segment_ids, 'start_pos': start_pos, 'end_pos': end_pos, 'tokens': tokens}

    def __iter__(self):

        dataset_iter = iter(self.dataset)
        steps_per_epoch = len(self.dataset)//self.batchsz
        for indices in range(steps_per_epoch):
            step = [next(dataset_iter) for _ in range(self.batchsz)]
            yield self._batch(step)



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

    def load(self, filename, vocabs, batchsz, **kwargs):

        #shuffle = kwargs.get('shuffle', False)
        doc_stride = int(kwargs.get('doc_stride', 128))
        mxqlen = int(kwargs.get('mxqlen', 64))
        has_impossible = bool(kwargs.get('has_impossible', True))
        # I think for both training and dev in MEAD execution we need this to be true
        is_training = bool(kwargs.get('is_training', True))

        loader = MRCDatasetIterator(filename, self.vectorizer, mxlen=384, doc_stride=doc_stride,
                                    has_impossible=has_impossible, is_training=is_training, mxqlen=mxqlen)

        return Batcher(loader, batchsz)


########################################################################################
# Training Utils


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

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

    def _test(self, loader, **kwargs):
        self.model.eval()
        total_loss = 0
        total_norm = 0
        null_thresh = kwargs.get('null_thresh', 0.5)
        steps = len(loader)
        pg = create_progress_bar(steps)

        exact_matches = Average('exact_match')
        precisions = Average('precision')
        recalls = Average('recall')
        f1s = Average('f1')
        for batch_dict in pg(loader):
            with torch.no_grad():
                example = self._make_input(batch_dict)
                y_start_pos = example.pop('start_pos')
                y_end_pos = example.pop('end_pos')
                start_pos_pred, end_pos_pred = self.model(example)
                loss = self.crit(start_pos_pred, end_pos_pred, y_start_pos, y_end_pos)
                batchsz = self._get_batchsz(example)
                total_loss += loss.item() * batchsz
                total_norm += batchsz
                best_end = torch.argmax(end_pos_pred, -1)
                for i in range(batchsz):
                    tokens = example['tokens'][i]
                    best_end_idx = best_end[i].item()


                    best_start_idx = torch.argmax(start_pos_pred[i], -1).item()
                    sum_preds = torch.sigmoid(start_pos_pred[i, best_start_idx]) + torch.sigmoid(end_pos_pred[i, best_end_idx])
                    if sum_preds < null_thresh:
                        pred_answer = ''
                    elif best_start_idx > best_end_idx:
                        pred_answer = ''
                    else:
                        pred_answer = ' '.join(tokens[best_start_idx:best_end_idx])
                    real_answer = ' '.join(tokens[y_start_pos[i].item():y_end_pos[i].item()])
                    exact_match = compute_exact(real_answer, pred_answer)
                    exact_matches.update(exact_match)
                    precision, recall, f1 = compute_f1(real_answer, pred_answer)
                    precisions.update(precision)
                    recalls.update(recall)
                    f1s.update(f1)


        metrics = {}
        metrics['avg_loss'] = total_loss / float(total_norm)
        metrics['precision'] = precisions.avg
        metrics['recall'] = recalls.avg
        metrics['f1'] = f1s.avg
        return metrics

    def _make_input(self, batch_dict):
        ex = {}
        for k, v in batch_dict.items():
            ex[k] = v.cuda() if k != 'tokens' else v
        return ex

    def _train(self, loader, **kwargs):
        self.model.train()
        reporting_fns = kwargs.get('reporting_fns', [])
        steps = len(loader)
        pg = create_progress_bar(steps)
        epoch_loss = 0
        epoch_div = 0
        for i, batch_dict in enumerate(pg(loader)):
            if i == 15000:
                break
            self.optimizer.zero_grad()
            example = self._make_input(batch_dict)
            y_start_pos = example.pop('start_pos')
            y_end_pos = example.pop('end_pos')
            start_pos_pred, end_pos_pred = self.model(example)
            loss = self.crit(start_pos_pred, end_pos_pred, y_start_pos, y_end_pos)
            batchsz = self._get_batchsz(example)
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

    num_loader_workers = int(kwargs.get('num_loader_workers', 0))
    pin_memory = bool(kwargs.get('pin_memory', True))
    ts = DataLoader(ts, num_workers=num_loader_workers, batch_size=None, pin_memory=pin_memory)
    vs = DataLoader(vs, batch_size=None, pin_memory=pin_memory)
    es = DataLoader(es, batch_size=None, pin_memory=pin_memory) if es is not None else None

    best_metric = 0
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'f1')
        early_stopping_cmp, best_metric = get_metric_cmp(early_stopping_metric, kwargs.get('early_stopping_cmp'))
        patience = kwargs.get('patience', epochs)
        logger.info('Doing early stopping on [%s] with patience [%d]', early_stopping_metric, patience)

    reporting_fns = listify(kwargs.get('reporting', []))
    logger.info('reporting %s', reporting_fns)
    trainer = create_trainer(model_params, **kwargs)

    last_improved = 0

    for epoch in range(epochs):
        trainer.train(ts, reporting_fns)
        test_metrics = trainer.test(vs, reporting_fns)

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
        ##return baseline.model.create_model(self.embeddings, self.labels, **model)

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
        )
        self.valid_data = self.reader.load(
            self.dataset['valid_file'],
            self.feat2index,
            vbsz,
        )
        self.test_data = None
        if 'test_file' in self.dataset:
            self.test_data = self.reader.load(
                self.dataset['test_file'],
                self.feat2index,
                tbsz,
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

class StartAndEndPlusImpossible(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_start = kwargs.get('weight_start', 1/3.)
        self.weight_end = kwargs.get('weight_end', 1/3.)
        self.weight_impossible = kwargs.get('impossible', 1/3.)
        self.loss_fct = nn.CrossEntropyLoss()
        self.impossible_loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, start_logits, end_logits, truth_start, truth_end, impossible_guess, truth_impossible):
        start_loss = self.loss_fct(start_logits, truth_start)
        end_loss = self.loss_fct(end_logits, truth_end)
        impossible_loss = self.impossible_loss_fct(impossible_guess, truth_impossible)
        total_loss = self.weight_start * start_loss + self.weight_end * end_loss + self.weight_impossible * impossible_loss
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
        """Create a tagger from the inputs.  Most classes shouldnt extend this

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