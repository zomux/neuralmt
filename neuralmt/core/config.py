#!/usr/bin/env python
# -*- coding: utf-8 -*-

class NeuralMTPath(object):

    def __init__(self, vocab, weight, encoder, decoder, expander, decoder_hidden_size=1000, input_path=None):
        self.input_path = input_path
        self.vocab = vocab
        self.vocab_size = len(open(vocab).readlines())
        self.weight = weight
        self.encoder = encoder
        self.decoder = decoder
        self.expander = expander
        self.hidden_size = decoder_hidden_size



class NeuralMTConfiguration(object):
    """
    Configuration of neural machine translation.
    """

    def __init__(self, target_vocab="", char_based=False, start_token="<s>", end_token="</s>"):
        self.target_vocab = target_vocab
        self.target_vocab_size = len(open(target_vocab).readlines())
        self.char_based = char_based
        self._paths = []
        self.end_token = end_token
        self.start_token = start_token

    def add_mt_path(self, vocab, mt_model, weight=1.0, input_path=None):
        """
        Add one encoder-decoder model to the ensemble.
        :type mt_model: EncoderDecoderModel
        """
        encoder, decoder, expander = mt_model.export_test_components()
        self._paths.append(NeuralMTPath(vocab, weight, encoder, decoder, expander, decoder_hidden_size=mt_model.decoder_hidden_size(), input_path=input_path))
        return self

    def add_path(self, vocab, encoder, decoder, expander, weight=1.0, decoder_hidden_size=1000, input_path=None):
        """
        Add one translation path to the ensemble.
        :return:
        """
        self._paths.append(NeuralMTPath(vocab, weight, encoder, decoder, expander, decoder_hidden_size=decoder_hidden_size, input_path=input_path))
        return self

    def paths(self):
        """
        Return translation paths.
        :rtype: list of NeuralMTPath
        """
        return self._paths
