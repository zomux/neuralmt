#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *

class TMCostLayer(NeuralLayer):

    def __init__(self, target, mask, target_size):
        """
        :param target: 2d (batch, time)
        :param mask:  2d (batch, time)
        :param target_size: scalar
        """
        super(TMCostLayer, self).__init__("tm_cost")
        self.target = target
        self.mask = mask
        self.target_size = target_size

    def output(self, x):
        """
        :param x: 3d tensor (batch, time, vocab)
        """
        flat_mask = self.mask.flatten()

        # Softmax
        shape = x.shape
        x = x.reshape((shape[0] * shape[1], shape[2])) * flat_mask[:, None]
        softmax_tensor = T.nnet.softmax(x)

        # Get cost
        result_vector = softmax_tensor.flatten()
        target_vector = self.target.flatten()
        target_index_vector =  T.arange(target_vector.shape[0]) * self.target_size + target_vector

        prob_vector = result_vector[target_index_vector]
        prob_vector = T.clip(prob_vector, EPSILON, 1.0 - EPSILON)
        log_prob_vector = - T.log(prob_vector) * flat_mask
        cost = T.sum(log_prob_vector) / T.sum(flat_mask)
        return cost
