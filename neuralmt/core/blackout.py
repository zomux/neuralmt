#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *

class BlackOutCost(NeuralLayer):

    def __init__(self, vocab_size, sample_size):
        super(BlackOutCost, self).__init__("blackout_cost")
        self.vocab_size = vocab_size
        self.sample_size = sample_size

    def prepare(self):
        self.W = self.create_weight(self.input_dim, self.vocab_size, "W")

    def compute_tensor(self, x, y, neg_y):
        """
        x: (time, batch, hidden_size)
        y: (time, batch)
        neg_y: (time, batch, sample_size)
        """
        W_c = self.W[y.flatten()].reshape(y.shape)
        