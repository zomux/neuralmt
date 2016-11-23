#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.import_all import *

class NeuralTM(ComputationalGraph):

    def __init__(self, input_dim, model=None, config=None, input_tensor=None, monitor_callback=None,
                 sampling=False, sampling_len=50):
        self.monitor_callback = monitor_callback
        self.sampling = sampling
        self.sampling_len = sampling_len
        super(NeuralTM, self).__init__(input_dim, model, config, input_tensor)

    def setup_variables(self):
        super(NeuralTM, self).setup_variables()
        if self.sampling:
            self.target_length = self.sampling_len
        else:
            self.target_matrix =  T.imatrix('y')
            self.target_mask =  T.matrix('mask')
            self.target_variables.extend([self.target_matrix, self.target_mask])
            self.target_length = self.target_matrix.shape[1]
            if self.monitor_callback:
                self.training_callbacks.append(self.monitor_callback)

    def decode(self, output_vec, vocab_map):
        """
        vocab map does not contain "<eol>"
        """
        tokens = []
        vocab_size = len(vocab_map)
        for n in output_vec:
            if n == vocab_size:
                break
            else:
                tokens.append(vocab_map[n])
        return tokens
