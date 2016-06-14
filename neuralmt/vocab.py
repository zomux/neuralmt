#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle

class NeuralVocab(object):
    """
    Vocab class.
    """

    def __init__(self, vocab_path=None, unk_token="UNK"):
        self.unk_token = unk_token
        if vocab_path:
            self.load(vocab_path)

    def load(self, path):
        self._vocab = pickle.load(open(path))
        self._vocab_map = {}
        for i, tok in enumerate(self._vocab):
            self._vocab_map[tok] = i

    def encode(self, tokens):
        return map(self.encode_token, tokens)

    def encode_token(self, token):
        if token in self._vocab_map:
            return self._vocab_map[token]
        else:
            return self._vocab_map[self.unk_token]

    def decode(self, indexes):
        return map(self.decode_token, indexes)

    def decode_token(self, index):
        return self._vocab[index] if index < len(self._vocab) else self.unk_token

    def contains(self, token):
        return token in self._vocab_map

    def size(self):
        return len(self._vocab)
