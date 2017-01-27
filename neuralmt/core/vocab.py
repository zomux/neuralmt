#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from collections import Counter

DEFAULT_SPECIAL_TOKENS = ["<null>", "<s>", "</s>", "UNK"]


class NeuralVocab(object):
    """
    Vocab class.
    """

    def __init__(self, vocab_path=None, unk_token="UNK"):
        self._vocab = []
        self._vocab_map = {}
        self._unk_token = unk_token
        if vocab_path:
            self.load(vocab_path)

    def build(self, txt_path, limit=None, special_tokens=None, delimiter=" "):
        vocab_counter = Counter()
        for line in open(txt_path).xreadlines():
            words = line.strip().split(delimiter)
            vocab_counter.update(words)
        if special_tokens is None:
            special_tokens = DEFAULT_SPECIAL_TOKENS
        if limit is not None:
            final_items = vocab_counter.most_common()[:limit - len(special_tokens)]
        else:
            final_items = vocab_counter.most_common()
        final_items.sort(key=lambda x: (-x[1], x[0]))
        final_words = [x[0] for x in final_items]
        self._vocab = special_tokens + final_words
        self._build_vocab_map()

    def add(self, token):
        self._vocab.append(token)
        self._vocab_map[token] = self._vocab.index(token)

    def save(self, path):
        pickle.dump(self._vocab, open(path, "wb"))

    def load(self, path):
        self._vocab = pickle.load(open(path))
        self._build_vocab_map()

    def _build_vocab_map(self):
        self._vocab_map = {}
        for i, tok in enumerate(self._vocab):
            self._vocab_map[tok] = i

    def encode(self, tokens):
        return map(self.encode_token, tokens)

    def encode_token(self, token):
        if token in self._vocab_map:
            return self._vocab_map[token]
        else:
            return self._vocab_map[self._unk_token]

    def decode(self, indexes):
        return map(self.decode_token, indexes)

    def decode_token(self, index):
        return self._vocab[index] if index < len(self._vocab) else self._unk_token

    def contains(self, token):
        return token in self._vocab_map

    def size(self):
        return len(self._vocab)

    def get_list(self):
        return self._vocab
