import re
from itertools import chain
from typing import List, Pattern
import numpy as np
from baseline.services import ClassifierService, ONNXClassifierService


PERIOD_REGEX = re.compile(r"\.")
QUOTE_REGEX = re.compile(r"^(?:'{1,2}|\")")
DEFAULT_CONTEXT = 40
DEFAULT_SEPARATOR = "<SEP>"
DEFAULT_SENTENCE_BREAK_CLASS = "1"


class SentenceSegmenterService(ClassifierService):
    def __init__(
        self,
        *args,
        context_size: int = DEFAULT_CONTEXT,
        separator: str = DEFAULT_SEPARATOR,
        sentence_break_class: str = DEFAULT_SENTENCE_BREAK_CLASS,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.context_size = context_size
        self.separator = separator
        self.sentence_break_class = sentence_break_class

    def predict(self, tokens: str, boundy_candidate_regex: Pattern = PERIOD_REGEX, quote_regex: Pattern = QUOTE_REGEX) -> List[str]:
        possible_splits = list(boundy_candidate_regex.finditer(tokens))
        if not possible_splits:
            return [tokens]
        batch = []
        for split in possible_splits:
            left_context = tokens[max(0, split.start() - self.context_size): split.start()]
            right_context = tokens[split.end():min(len(tokens), split.end() + self.context_size)]
            example = list(chain(left_context.split(), (self.separator,), right_context.split()))
            batch.append(example)
        self.prepare_vectorizers(batch)
        examples = self.vectorize(batch)
        preds = self.model.predict(examples)
        preds = self.format_output(preds)
        sentences = []
        offset = 0
        for split, pred in zip(possible_splits, preds):
            if pred[0][0] == self.sentence_break_class:
                post = tokens[split.end():]
                quote = quote_regex.match(post)
                if quote:
                    end = split.end() + quote.end()
                else:
                    end = split.end()
                sentences.append(tokens[offset: end])
                offset = end
        if offset < len(tokens):
            sentences.append(tokens[offset:])
        return sentences

    @classmethod
    def load(cls, bundle, **kwargs):
        backend = kwargs.get('backend', 'tf')
        if backend == 'onnx':
            return ONNXSentenceSegmenterService.load(bundle, **kwargs)
        return super().load(bundle, **kwargs)


class ONNXSentenceSegmenterService(ONNXClassifierService):
    def __init__(
        self,
        *args,
        context_size: int = DEFAULT_CONTEXT,
        separator: str = DEFAULT_SEPARATOR,
        sentence_break_class: str = DEFAULT_SENTENCE_BREAK_CLASS,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.context_size = context_size
        self.separator = separator
        self.sentence_break_class = sentence_break_class

    def predict(self, tokens: str, boundy_candidate_regex: Pattern = PERIOD_REGEX, quote_regex: Pattern = QUOTE_REGEX) -> List[str]:
        possible_splits = list(boundy_candidate_regex.finditer(tokens))
        if not possible_splits:
            return [tokens]
        batch = []
        for split in possible_splits:
            left_context = tokens[max(0, split.start() - self.context_size): split.start()]
            right_context = tokens[split.end():min(len(tokens), split.end() + self.context_size)]
            example = list(chain(left_context.split(), (self.separator,), right_context.split()))
            batch.append(example)
        self.prepare_vectorizers(batch)
        examples = [self.vectorize([b]) for b in batch]
        preds = np.concatenate([self.model.run(None, example)[0] for example in examples])
        preds = self.format_output(preds)
        sentences = []
        offset = 0
        for split, pred in zip(possible_splits, preds):
            if pred[0][0] == self.sentence_break_class:
                post = tokens[split.end():]
                quote = quote_regex.match(post)
                if quote:
                    end = split.end() + quote.end()
                else:
                    end = split.end()
                sentences.append(tokens[offset: end])
                offset = end
        if offset < len(tokens):
            sentences.append(tokens[offset:])
        return sentences
