#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *

class OldSoftAttentionalLayer(NeuralLayer):

    def __init__(self, recurrent_unit, steps=40, mask=None, predict_input=None, test=False):
        """
        :type recurrent_unit: NeuralLayer
        :param test: indicate if this is for the test time.
        """
        super(OldSoftAttentionalLayer, self).__init__("tm_search")
        self.recurrent_unit = recurrent_unit
        self.steps = steps
        self.mask = mask.tensor if type(mask) == NeuralVariable else mask
        self.predict_input = predict_input.tensor if type(predict_input) == NeuralVariable else predict_input
        self.test = test

    def prepare(self):
        """
        Initialize the parameters, they are named following the original paper.
        """
        self.recurrent_unit.initialize(self.input_dim)
        self.register_inner_layers(self.recurrent_unit)
        self.output_dim = self.recurrent_unit.output_dim

        self.Ua = self.create_weight(self.input_dim, self.output_dim, "ua")
        self.Wa = self.create_weight(self.output_dim, self.output_dim, "wa")
        self.Va = self.create_weight(suffix="va", shape=(self.output_dim,))
        self.register_parameters(self.Va, self.Wa, self.Ua)
        self.tanh = build_activation('tanh')

    def _align(self, s_prev, UaH):
        """
        :param s_prev: (batch, output_dim)
        :param x: (batch, time, input_dim)
        :param UaH: T.dot(x, Ua) (batch, time, output_dim)
        :return: (batch, time)
        """
        WaSp = T.dot(s_prev, self.Wa)
        # For test time the UaH will be (time, output_dim)
        if self.test:
            preact = WaSp[:, None, :] + UaH[None, :, :]
        else:
            preact = WaSp[:, None, :] + UaH
        act = self.tanh(preact)
        aligns = T.dot(act, self.Va)
        if self.mask:
            if self.test:
                aligns *= self.mask[None, :]
            else:
                aligns *= self.mask
        aligns = T.nnet.softmax(aligns)
        return aligns

    def step(self, *vars):
        """
        vars: s_prev (c_prev) x UaH

        :param s_prev: (batch, output_dim)
        :param x: (batch, time, input_dim)
        :param UaH: (batch, time, output_dim)
        """
        if self.predict_input:
            predict_input = vars[0]
            vars = vars[1:]
        else:
            predict_input = None
        x, UaH = vars[-2:]
        s_prev = vars[0]
        align_weights = self._align(s_prev, UaH) # (batch, time)
        context_matrix = T.sum(align_weights[:, :, None] * x, axis=1) # (batch, input_dim)
        rnn_input_vars = self.recurrent_unit.produce_input_sequences(context_matrix, second_input=predict_input) + list(vars[:-2])
        new_s = self.recurrent_unit.step(*rnn_input_vars)
        return new_s

    def compute_tensor(self, x):
        """
        :param x: 3d tensor (batch, time, hidden_size x 2)
        """
        init_states = self.recurrent_unit.produce_initial_states(x)
        UaH = T.dot(x, self.Ua) # (batch, time, output_dim)
        sequences = [self.predict_input] if self.predict_input else None
        outputs, _ = theano.scan(self.step, sequences=sequences, outputs_info=init_states, non_sequences=[x, UaH],
                                 n_steps=self.steps)
        if type(outputs) == list:
            outputs = outputs[0]
        return outputs.dimshuffle((1, 0, 2))
