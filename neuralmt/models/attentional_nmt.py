#!/usr/bin/env python
# -*- coding: utf-8 -*-

import deepy as D
import deepy.layers as L
import deepy.tensor as T
from encoder_decoder import EncoderDecoderModel

class AttentionalNMT(EncoderDecoderModel):

    def prepare(self):
        self.src_embed_layer = L.WordEmbedding(self._embed_size, self._src_vocab_size)
        self.tgt_embed_layer = L.WordEmbedding(self._embed_size, self._tgt_vocab_size)
        self.forward_encoder = L.LSTM(self._hidden_size)
        self.backward_encoder = L.LSTM(self._hidden_size)
        self.first_state_nn = L.Dense(self._hidden_size, 'tanh')
        self.decoder_rnn = L.LSTM(self._hidden_size)
        self.attention = L.Attention(input_dim=self._hidden_size * 2, hidden_size=self._hidden_size)
        if self._hidden_size >= 1000:
            # Approx layer
            self.expander_nn = L.Chain(L.Dense(600), L.Dense(self._tgt_vocab_size))
        else:
            self.expander_nn = L.Dense(self._tgt_vocab_size)
        self._layers = [
            self.src_embed_layer, self.tgt_embed_layer,
            self.forward_encoder, self.backward_encoder,
            self.first_state_nn,
            self.decoder_rnn, self.attention,
            self.expander_nn
        ]

    def encode(self, input_vars, input_mask=None):
        input_embeds = self.src_embed_layer.compute(input_vars, mask=input_mask)

        # Encoder
        forward_rnn_var = self.forward_encoder.compute(input_embeds, mask=input_mask)
        backward_rnn_var = T.reverse(self.backward_encoder.compute(input_embeds, mask=input_mask, backward=True), axis=1)
        encoder_states = T.concat([forward_rnn_var, backward_rnn_var], axis=2)
        precomputed_att_values = self.attention.precompute(encoder_states)

        return {
            "encoder_states": encoder_states,
            "init_state": self.first_state_nn.compute(backward_rnn_var[:, 0]),
            "precomputed_values": precomputed_att_values
        }

    def decode_step(self, vars):

        context_vector = self.attention.compute_context_vector(vars.state, vars.encoder_states,
                                                               precomputed_values=vars.precomputed_values,
                                                               mask=vars.input_mask)
        decoder_input = T.concat([context_vector, vars.feedback])
        new_state, new_cell = self.decoder_rnn.compute_step(vars.state, lstm_cell=vars.cell, input=decoder_input)
        vars.state = new_state
        vars.cell = new_cell

    def lookup_feedback(self, feedback):
        return self.tgt_embed_layer.compute(feedback)

    def expand(self, decoder_outputs):
        return self.expander_nn.compute(decoder_outputs.state)