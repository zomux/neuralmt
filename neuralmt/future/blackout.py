#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy.import_all import *

class BlackOutCost(L.NeuralLayer):

    def __init__(self, vocab_size, sample_size, word_dist):
        super(BlackOutCost, self).__init__("blackout_cost")
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.word_dist = theano.shared(word_dist)
        self.word_dict = self._generate_word_dict(word_dist)
        self.dict_len = self.word_dict.get_value().shape[0]

    def _generate_word_dict(self, word_dist):
        dict_array = []
        for i, prob in enumerate(word_dist):
            dict_array.extend([i]*(int(1000*prob + 1)))
        return theano.shared(np.array(dict_array))

    def prepare(self):
        self.W = self.create_weight(self.vocab_size, self.input_dim, "W")
        self.B = self.create_bias(self.vocab_size, "B")
        self.register_parameters(self.W, self.B)

    def compute_tensor(self, x, y, y_mask):
        """
        x: (time, batch, hidden_size)
        y: (time, batch)
        y_mask: (time, batch)
        """
        flat_dim = y.shape[0] * y.shape[1]
        flat_y = y.reshape((flat_dim,))
        flat_x = x.reshape((flat_dim, -1))
        flat_mask = y_mask.reshape((flat_dim,))

        output_m = T.dot(flat_x, self.W.T) + self.B # - (time*batch, vocab)

        output_vec = output_m.flatten()

        # Sample negative words
        neg_samp_ids = T.cast((D.env.global_theano_rand.uniform((self.sample_size,)) * self.dict_len), 'int32')  # - (time*batch,)
        neg_samp_tokens = self.word_dict[neg_samp_ids]
        neg_samp_mask = T.neq(neg_samp_tokens, flat_y.reshape((flat_dim, 1)))  # - (time*batch, sample_size)

        # Positive output values
        flat_y_off = T.arange(flat_dim) * self.vocab_size + flat_y
        z_pos = output_vec[flat_y_off]  # - (time*batch,)

        # Negative output values
        neg_off = neg_samp_tokens + T.arange(flat_dim).reshape((-1, 1)) * self.vocab_size
        z_neg = output_vec[neg_off]

        # BlackOut Cost
        ep_pos = self.word_dist[flat_y] * T.exp(z_pos)  # - (time*batch,)
        ep_neg = self.word_dist[neg_samp_tokens] * T.exp(z_neg) * neg_samp_mask  # - (time*batch, sample_size)

        nominator = ep_pos + T.sum(ep_neg, axis=1) + D.env.EPSILON
        p_pos = ep_pos / nominator
        p_neg = ep_neg / nominator.dimshuffle(0, 'x')

        cost_vec = T.log(p_pos) + T.sum(T.log(1 - p_neg), axis=1)
        cost = T.sum(cost_vec * flat_mask) / T.sum(flat_mask)
        return -cost

    def compute_tensor(self, x, y, y_mask):
        """
        x: (time, batch, hidden_size)
        y: (time, batch)
        y_mask: (time, batch)
        """
        flat_dim = y.shape[0] * y.shape[1]
        flat_y = y.reshape((flat_dim,))
        flat_x = x.reshape((flat_dim, -1))
        flat_mask = y_mask.reshape((flat_dim,))

        output_m = T.dot(flat_x, self.W.T)  # - (time*batch, vocab)

        softmax_m = T.nnet.softmax(output_m)

        # Get cost
        result_vector = softmax_m.flatten()
        target_index_vector = T.arange(flat_y.shape[0]) * self.vocab_size + flat_y

        prob_vector = result_vector[target_index_vector]
        prob_vector = T.clip(prob_vector, D.env.EPSILON, 1.0 - D.env.EPSILON)
        log_prob_vector = - T.log(prob_vector) * flat_mask
        cost = T.sum(log_prob_vector) / T.sum(flat_mask)
        return cost