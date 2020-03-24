import collections
from baseline.utils import Offsets
from baseline.vectorizers import Char1DVectorizer, register_vectorizer


@register_vectorizer(name="sent-seg-context-window-sep")
class ContextWindowChar1DVectorizer(Char1DVectorizer):
    SEP = "<SEP>"

    def _next_element(self, tokens, vocab):
        OOV = vocab['<UNK>']
        EOW = vocab.get('<EOW>', vocab.get(' ', Offsets.PAD))
        for token in self.iterable(tokens):
            if token == ContextWindowChar1DVectorizer.SEP:
                yield vocab['<SEP>']
                continue
            for ch in token:
                yield vocab.get(ch, OOV)
            yield EOW

    def count(self, tokens):
        seen_tok = 0
        counter = collections.Counter()
        for token in self.iterable(tokens):
            seen_tok += 1
            if token == ContextWindowChar1DVectorizer.SEP:
                counter[ContextWindowChar1DVectorizer.SEP] += 1
                counter['<EOW>'] += 1
                seen_tok += 1
                continue
            for ch in token:
                counter[ch] += 1
                seen_tok += 1
            counter['<EOW>'] += 1
            seen_tok += 1

        self.max_seen_tok = max(self.max_seen_tok, seen_tok)
        return counter
