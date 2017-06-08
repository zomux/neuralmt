#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from neuralmt import EncoderDecoderModel
from deepy.utils import MapDict
import copy

class BeamSearchKit(object):

    def __init__(self, model, source_vocab, target_vocab, start_token="<s>", end_token="</s>", beam_size=5, opts=None):
        assert isinstance(model, EncoderDecoderModel)
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.start_token = start_token
        self.end_token = end_token
        self.start_token_id = self.source_vocab.encode_token(start_token)
        self.end_token_id = self.target_vocab.encode_token(end_token)
        self.opts = MapDict(opts) if opts else opts
        self.beam_size = beam_sizet
        self.prepare()
        
    def prepare(self):
        """
        A prepraration function.
        """
        
    def preprocess(self, sentence):
        return self.source_vocab.encode(sentence.split())
    
    def postprocess(self, input, raw_result):
        return self.target_vocab.decode(raw_result)
    
    def translate(self, sentence):
        """
        Translate one sentence.
        :return: result, score
        """
        
        input_tokens = self.preprocess(sentence)
        result, score = self.beam_search(input_tokens)
        # Special case
        if result:
            result_words = self.postprocess(sentence, result)
            if not result_words:
                result_words.append("EMPTY")
            output_line = " ".join(result_words)
            return output_line, score
        else:
            return None, None
        
        
    def init_hyps(self, init_state=None, items=None):
        final_hyps = []
        # hyp: state, tokens, sum of -log
        state = np.zeros((self.model.decoder_hidden_size(),), dtype="float32")
        if init_state is not None:
            state[:init_state.shape[0]] = init_state
        first_hyp = {
            "state": state,
            "tokens": [self.start_token_id],
            "logp": 0.
        }
        if items:
            first_hyp.update(items)
        hyps = [first_hyp]
        return hyps, final_hyps
    
    
    def expand_hyps(self, hyps, batch_new_states, batch_scores):
        """
        Create B x B new hypotheses
        """
        new_hyps = []
        for i, hyp in enumerate(hyps):
            new_state = batch_new_states[i]
            logprob = batch_scores[i] + hyp["logp"]
            best_indices = sorted(np.argpartition(logprob, self.beam_size)[:self.beam_size], key=lambda x: logprob[x])
            for idx in best_indices:
                new_hyp = {
                    "state": new_state,
                    "tokens": hyp["tokens"] + [idx],
                    "logp": logprob[idx],
                    "old_state": hyp["state"]
                }
                # Keep old information
                for key in hyp:
                    if key not in new_hyp:
                        new_hyp[key] = copy.copy(hyp[key])
                new_hyps.append(new_hyp)
        new_hyps.sort(key=lambda h: h["logp"])
        return new_hyps
    
    def truncate_hyps(self, new_hyps, final_hyps=None):
        """
        Collect finished hyps and truncate.
        """
        # Get final hyps
        if final_hyps:
            for i in range(len(new_hyps)):
                hyp = new_hyps[i]
                if hyp["tokens"][-1] == self.end_token_id:
                    tokens = hyp["tokens"][1:-1]
                    final_hyps.append({
                        "tokens": tokens,
                        "logp": hyp["logp"] / len(tokens),
                        "raw": hyp
                    })
        # Update hyps
        hyps = [h for h in new_hyps if h["tokens"][-1] != self.end_token_id][:self.beam_size]
        return hyps, final_hyps
    
    def update_hyps(self, hyps, final_hyps, batch_new_states, batch_scores):
        """
        Expand and Truncate hypotheses.
        """
        new_hyps = self.expand_hyps(hyps, batch_new_states, batch_scores)
        hyps, final_hyps = self.truncate_hyps(new_hyps, final_hyps)
        return hyps, final_hyps
    
    def beam_search(self, input_tokens):
        raise NotImplementedError
        return None, None
    
    
