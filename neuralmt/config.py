#!/usr/bin/env python
# -*- coding: utf-8 -*-

class NeuralMTPath(object):

    def __init__(self, input_path, model, vocab, vocab_size, weight):
        self.input_path = input_path
        self.model = model
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.weight = weight


class NeuralMTConfiguration(object):
    """
    Configuration of neural machine translation.
    """

    def __init__(self, target_vocab="",
                 target_vocab_size=10000,
                 hidden_size=1000,
                 word_embed=1000,
                 approx_size=600,
                 arch="lstm_one_layer_search",
                 char_based=False):
        self.target_vocab = target_vocab
        self.target_vocab_size = target_vocab_size
        self.hidden_size = hidden_size
        self.word_embed = word_embed
        self.arch = arch
        self.char_based = char_based
        self.approx_size = approx_size
        self._paths = []

    def add_path(self, input_path, model, vocab, vocab_size, weight=1.0):
        """
        Add one model (translation path) to the ensemble.
        :return:
        """
        self._paths.append(NeuralMTPath(input_path, model, vocab, vocab_size, weight))
        return self

    def paths(self):
        """
        Return translation paths.
        :rtype: list of NeuralMTPath
        """
        return self._paths
