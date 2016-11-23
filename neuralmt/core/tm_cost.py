#!/usr/bin/env python
# -*- coding: utf-8 -*-

import deepy as D
import deepy.tensor as T
import deepy.layers as L

class TMCostLayer(L.NeuralLayer):

    def __init__(self, target, mask, target_size, cost_map = None):
        """
        :param target: 2d (batch, time)
        :type target: NeuralVariable
        :param mask:  2d (batch, time)
        :type mask: NeuralVariable
        :param target_size: scalar
        """
        super(TMCostLayer, self).__init__("tm_cost")
        self.target = target.tensor
        self.mask = mask.tensor
        self.target_size = target_size
        self.cost_map = cost_map

    def compute_tensor(self, x, without_softmax=False):
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
        prob_vector = T.clip(prob_vector, D.env.EPSILON, 1.0 - D.env.EPSILON)
        log_prob_vector = - T.log(prob_vector) * flat_mask
        if self.cost_map:
            log_prob_vector *= self.cost_map.flatten()
        cost = T.sum(log_prob_vector) / T.sum(flat_mask)
        return cost
