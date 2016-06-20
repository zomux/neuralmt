#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *

class ContextDeconv(NeuralLayer):

    def __init__(self, hidden_size=100):
        super(ContextDeconv, self).__init__('context_deconv')
        self._size = hidden_size

    def prepare(self):
        self.output_dim = self._size
        self.core_W = self.create_weight(self._size, self._size * 2, "size")
        self.start_W = self.create_weight(self.input_dim, self._size)
        self.start_B = self.create_bias(self._size)

    def step(self, varmap):
        full_nodes = varmap['nodes']  # - (batch, nodes, size)
        step = varmap['step']
        batch_size = varmap['batch_size']
        nodes_len = batch_size * T.power(2, step)
        nodes = full_nodes[:nodes_len, :]
        new_nodes = T.dot(nodes, self.core_W)  # - (batch*nodes, size*2)
        new_nodes = new_nodes.reshape((nodes.shape[0] * 2, self._size))
        return {'nodes': T.set_subtensor(full_nodes[:nodes.shape[0] * 2, :], new_nodes),
                'step': step + 1}

    def compute_tensor(self, x, n_steps):
        """
        x: (batch, input_dim)
        """
        batch_size = x.shape[0]
        full_nodes = T.zeros((batch_size * T.power(2, n_steps), self._size), dtype=FLOATX)
        start_node = T.dot(x, self.start_W) + self.start_B
        output_map, _ = Scanner(self.step,
                             outputs_info={
                                 'nodes': T.set_subtensor(full_nodes[:batch_size, :], start_node),
                                 'step': 0,
                             },
                             non_sequences={
                                 'batch_size': batch_size
                             },
                             n_steps=n_steps).compute()
        nodes = output_map['nodes'].reshape((batch_size, -1, self._size))  # - (batch, times, size)
        return nodes