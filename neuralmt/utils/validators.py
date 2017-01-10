#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from bleu import smoothed_bleu
from deepy.trainers import TrainingValidator

class SimpleBleuValidator(TrainingValidator):
    """
    A simplified BLEU validator.
    """

    def __init__(self, valid_model, freq=1500, save_path=None):
        super(SimpleBleuValidator, self).__init__(valid_model, 'valid', freq=freq, save_path=save_path, criteria='bleu', smaller_is_better=False)

    def run(self, data_x):
        output_vars = self.compute(*data_x)
        _, _, tgt_tokens, tgt_masks = data_x
        bleus = []
        for i in range(tgt_tokens.shape[0]):
            target_len = int(tgt_masks[i].sum())
            ref_tokens = tgt_tokens[i, :target_len]
            out_tokens = output_vars.outputs[i, :target_len]
            bleus.append(smoothed_bleu(out_tokens, ref_tokens))
        output_vars.bleu = numpy.mean(bleus)
        return self._extract_costs(output_vars)