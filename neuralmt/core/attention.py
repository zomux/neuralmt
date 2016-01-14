#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *


class SoftAttentionalLayer(NeuralLayer):

    def __init__(self, recurrent_unit, exponential_weights=False, test=False, sample_feedback=None, initial_feedback=None):
        """
        :type recurrent_unit: RecurrentLayer
        :param test: indicate if this is for the test time.
        """
        super(SoftAttentionalLayer, self).__init__("attention")
        self.recurrent_unit = recurrent_unit
        self.test = test
        self.exponential_weights = exponential_weights
        self._sample_feedback = sample_feedback
        self._initial_feedback = initial_feedback

    def prepare(self):
        """
        Initialize the parameters, they are named following the original paper.
        """
        self.recurrent_unit.connect(self.input_dim)
        self.register_inner_layers(self.recurrent_unit)
        self.output_dim = self.recurrent_unit.output_dim

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
            if self.test:
                aligns *= mask[None, :]
            else:
                aligns *= mask
        aligns = T.nnet.softmax(aligns)
        return aligns

    def step(self, step_inputs):
        """
        :type step_inputs: dict
        s_prev: (batch, output_dim)
        x: (batch, time, input_dim)
        UaH: (batch, time, output_dim)
        """
        feedback, s_prev, UaH, mask, inputs = map(step_inputs.get, ["feedback", self.recurrent_unit.main_state, "UaH", "mask", "inputs"])

        align_weights = self._align(s_prev, UaH, mask) # (batch, time)
        context_matrix = T.sum(align_weights[:, :, None] * inputs, axis=1) # (batch, input_dim)
        recurrent_inputs = self.recurrent_unit.get_step_inputs(context_matrix, additional_inputs=[feedback], states=step_inputs)
        step_outputs = self.recurrent_unit.step(recurrent_inputs)
        if self._sample_feedback:
            new_feedback = self._sample_feedback(step_outputs[self.recurrent_unit.main_state])
            step_outputs["feedback"] = new_feedback

        return step_outputs

    def compute_pre_states(self, inputs):
        """
        :type inputs: NeuralVar
        :return: NeuralVar
        """
        return {"UaH", inputs.apply(lambda t: T.dot(t, self.Ua), dim=self.output_dim)}

    def output(self, inputs, mask=None, feedback=None, steps=None):
        """
        :param inputs: 3d tensor (batch, time, hidden_size x 2 in bi-directional encoder)
        """
        init_state_map = self.recurrent_unit.get_initial_states(inputs)
        sequences = {
            "feedback": feedback
        }
        non_sequences = {
            "inputs": inputs,
            "UaH": T.dot(inputs, self.Ua), # (batch, time, output_dim)
            "mask": mask
        }
        if self._sample_feedback:
            sequences.clear()
            init_state_map["feedback"] = self._initial_feedback
        output_dict, updates = Scanner(self.step, sequences=sequences, outputs_info=init_state_map, non_sequences=non_sequences,
                                 n_steps=steps).compute()
        self.register_updates(*updates.items())
        outputs = output_dict["state"]
        return outputs.dimshuffle((1, 0, 2))
