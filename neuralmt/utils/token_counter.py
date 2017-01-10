#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter

import numpy as np

from neuralmt.core.vocab import NeuralVocab

def make_token_distribution(vocab_path, filepath):
    token_counter = Counter()
    vocab = NeuralVocab(vocab_path=vocab_path)
    for line in open(filepath).xreadlines():
        tokens = line.strip().split(" ")
        tokens.append("</s>")
        token_counter.update(vocab.encode(tokens))
    vocab_count = np.array([token_counter.get(i, 1) for i in range(vocab.size())], dtype='float32')
    dist = vocab_count / vocab_count.sum()
    return dist
