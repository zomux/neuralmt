#!/usr/bin/env python
# -*- coding: utf-8 -*-


from deepy import *


class SoftAttentionalLayer(NeuralLayer):

    def __init__(self, recurrent_unit, exponential_weights=False, feedback_callback=None, initial_feedback=None, sampling=False):
        """
        :type recurrent_unit: RecurrentLayer
        :param test: indicate if this is for the test time.
        """
        super(SoftAttentionalLayer, self).__init__("attention")
        self.rnn = recurrent_unit
        self.exponential_weights = exponential_weights
        self._feedback_callback = feedback_callback
        self._initial_feedback = initial_feedback
        self._sampling = sampling

    def prepare(self):
        """
        Initialize the parameters, they are named following the original paper.
        """
        self.rnn.initialize(self.input_dim)
        self.register_inner_layers(self.rnn)
        self.output_dim = self.rnn.output_dim

        self.Ua = self.create_weight(self.input_dim, self.output_dim, "ua")
        self.Wa = self.create_weight(self.output_dim, self.output_dim, "wa")
        self.Va = self.create_weight(suffix="va", shape=(self.output_dim,))
        self.register_parameters(self.Va, self.Wa, self.Ua)
        self.tanh = build_activation('tanh')

    def _align(self, s_prev, UaH, mask=None):
        """
        :param s_prev: (batch, output_dim)
        :param x: (batch, time, input_dim)
        :param UaH: T.dot(x, Ua) (batch, time, output_dim)
        :return: (batch, time)
        """
        WaSp = T.dot(s_prev, self.Wa)
        # For test time the UaH will be (time, output_dim)
        if UaH.ndim == 2:
            preact = WaSp[:, None, :] + UaH[None, :, :]
        else:
            preact = WaSp[:, None, :] + UaH
        act = self.tanh(preact)
        aligns = T.dot(act, self.Va) # ~ (batch, time)
        if self.exponential_weights:
            aligns = T.exp(aligns - T.max(aligns, axis=1)[None, :].T)
            aligns += EPSILON
        if mask:
            if aligns.ndim == 3:
                aligns *= mask[None, :]
            else:
                aligns *= mask
        aligns = T.nnet.softmax(aligns)
        return aligns

    @neural_computation
    def step(self, step_inputs):
        """
        :type step_inputs: dict
        s_prev: (batch, output_dim)
        x: (batch, time, input_dim)
        UaH: (batch, time, output_dim)
        """
        feedback, s_prev, UaH, mask, inputs = map(step_inputs.get, ["feedback", self.rnn.main_state, "UaH", "mask", "inputs"])

        align_weights = self._align(s_prev, UaH, mask) # (batch, time)
        if self._sampling:
            align_weights = global_theano_rand.multinomial(pvals=align_weights, dtype='float32')
        context_matrix = T.sum(align_weights[:, :, None] * inputs, axis=1) # (batch, input_dim)
        additional_inputs = [feedback] if feedback else []
        recurrent_inputs = self.rnn.get_step_inputs(context_matrix, additional_inputs=additional_inputs, states=step_inputs)
        step_outputs = self.rnn.step(recurrent_inputs)
        if self._feedback_callback:
            new_feedback = self._feedback_callback(step_outputs[self.rnn.main_state])
            step_outputs["feedback"] = new_feedback

        return step_outputs

    @neural_computation
    def get_step_inputs(self, input_var, state=None, feedback=None, states=None, **kwargs):
        state_map = {"inputs": input_var, "state": state, "feedback": feedback}
        if states:
            state_map.update(states)
        state_map.update(self.merge_inputs(input_var))
        state_map.update(kwargs)
        return state_map

    @neural_computation
    def merge_inputs(self, inputs):
        """
        :type inputs: NeuralVariable
        :return: NeuralVar
        """
        return {"UaH": T.dot(inputs, self.Ua)}

    @neural_computation
    def set_feedback_callback(self, func, initial_feedback):
        """
        Set feedback callback.
        :param func: a tensor function
        :param initial_feedback:
        :return:
        """
        self._feedback_callback = func
        self._initial_feedback = initial_feedback

    def compute_tensor(self, inputs, mask=None, feedback=None, steps=None):
        """
        :param inputs: 3d tensor (batch, time, hidden_size x 2 in bi-directional encoder)
        """
        init_state_map = self.rnn.get_initial_states(inputs)
        sequences = {
            "feedback": feedback
        }
        non_sequences = {
            "inputs": inputs,
            "UaH": T.dot(inputs, self.Ua), # (batch, time, output_dim)
            "mask": mask
        }
        if self._feedback_callback:
            sequences.clear()
            init_state_map["feedback"] = self._initial_feedback
        output_dict, updates = Scanner(self.step, sequences=sequences, outputs_info=init_state_map, non_sequences=non_sequences,
                                 n_steps=steps).compute()
        self.register_updates(*updates.items())
        outputs = output_dict["state"]
        return outputs.dimshuffle((1, 0, 2))
