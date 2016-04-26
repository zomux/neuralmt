#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from bleu import smoothed_bleu
from logging import getLogger


class BLEUValidator(object):
    """
    Monitor the BLEU scores of the valid data, and greedy optimize the mean BLEU score.
    """

    def __init__(self, valid_data, test_model, model_path, frequency=1500, end_token=2, patience=-1):
        """
        Initialize the BLEU validator.
        :param valid_data:
        :param test_model:
        :param model_path:
        :param frequency: number of iterations to validate the BLEU
        :param end_token:
        :param patience: with a negative patience, rolling back is disabled
        :return:
        """
        self._valid_data = valid_data
        self.model = test_model
        self._model_path = model_path
        self._counter = 0
        self._frequency = frequency
        self._best_bleu = 0
        self._end_token = 2
        self._logger = getLogger("BLEUValidator")
        self._checkpoint = None
        self._patience = patience
        self._current_failures = 0

    def __call__(self, trainer):
        if self._counter % self._frequency == 0:
            trainer._run_valid(-1, self._valid_data, dry_run=True)
            # Compute BLEUs
            bleus = []
            for x_batch, mask_batch, ref_batch, _ in self._valid_data:
                hyp_batch = self.model.compute(x_batch, mask_batch, ref_batch.shape[1])
                for hyp, ref in zip(hyp_batch, ref_batch):
                    bleus.append(self.bleu(hyp, ref))
            mean_bleu = np.mean(bleus)
            self._logger.info("BLEU: {:.2f}, Last best: {:.2f}".format(mean_bleu, self._best_bleu))
            # Make a checkpoint if a better BLEU is recorded
            if mean_bleu > self._best_bleu:
                self._best_bleu = mean_bleu
                self._checkpoint = trainer.copy_params()
                trainer.save_params(self._model_path)
                self._current_failures = 0
            else:
                self._current_failures += 1
            # Roll back to the last checkpoint if too many failures
            if self._patience >=0 and self._current_failures >= self._patience and self._checkpoint:
                self._logger.info("Roll back parameters to the last checkpoint")
                trainer.set_params(*self._checkpoint)
        self._counter += 1

    def bleu(self, hyp, ref):
        hyp = list(hyp)
        ref= list(ref)
        if self._end_token in hyp:
            hyp = hyp[:hyp.index(self._end_token)]
        if self._end_token in ref:
            ref = ref[:ref.index(self._end_token)]
        return smoothed_bleu(hyp, ref)
