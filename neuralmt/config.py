#!/usr/bin/env python
# -*- coding: utf-8 -*-

class NeuralMTPath(object):

    def __init__(self, input_path, vocab, vocab_size, weight, encoder, decoder, expander):
        self.input_path = input_path
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.weight = weight
        self.encoder = encoder
        self.decoder = decoder
        self.expander = expander


class NeuralMTConfiguration(object):
    """
    Configuration of neural machine translation.
    """

    def __init__(self, target_vocab="", target_vocab_size=10000, hidden_size=500,
                 char_based=False, start_token="<s>", end_token="</s>"):
        self.hidden_size = hidden_size
        self.target_vocab = target_vocab
        self.target_vocab_size = target_vocab_size
        self.char_based = char_based
        self._paths = []
        self.end_token = end_token
        self.start_token = start_token

    def add_path(self, input_path, vocab, vocab_size, encoder, decoder, expander, weight=1.0):
        """
        Add one model (translation path) to the ensemble.
        :return:
        """
        self._paths.append(NeuralMTPath(input_path, vocab, vocab_size, weight, encoder, decoder, expander))
        return self

    def paths(self):
        """
        Return translation paths.
        :rtype: list of NeuralMTPath
        """
        return self._paths
