#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging as loggers
import pickle

import numpy as np

from neuralmt.core.config import NeuralMTConfiguration, NeuralMTPath

logging = loggers.getLogger(__name__)

class NeuralMTComponent(object):

    def __init__(self, mt_path, config):
        """
        :type mt_path: NeuralMTPath
        :type config: NeuralMTConfiguration
        """
        self.mt_path = mt_path
        self.config = config
        self._build()

    def _build(self):
        self.hidden_size = self.mt_path.hidden_size
        self.encoder, self.decoder, self.expander = self._get_neural_components(self.mt_path)
        self.vocab = pickle.load(open(self.mt_path.vocab))
        self.vocab_map = {}
        for i, word in enumerate(self.vocab):
            self.vocab_map[word] = i
        if self.mt_path.input_path:
            self.inputs = map(str.strip, open(self.mt_path.input_path).readlines())
            self.input_tokens = map(self.get_tokens, self.inputs)
        else:
            self.inputs = []
            self.input_tokens = []

    def get_tokens(self, line):
        if isinstance(line, str):
            if self.config.char_based:
                src_words = list(line)
            else:
                src_words = line.split(" ")
        else:
            src_words = line
        # src_words.insert(0, "<s>")
        src_tokens = []
        for w in src_words:
            if w in self.vocab_map:
                src_tokens.append(self.vocab_map[w])
            else:
                src_tokens.append(self.vocab_map["UNK"])
        return np.array(src_tokens, dtype="int32")

    def _get_neural_components(self, mt_path):
        """
        Get neural encoder, decoder and expander.
        :type mt_path: NeuralMTPath
        """

        return mt_path.encoder, mt_path.decoder, mt_path.expander


class NeuralTranslator(object):

    def __init__(self, config):
        """
        :type config: NeuralMTConfiguration
        """
        self.config = config
        self._prepare()

    def _split(self, sent):
        """
        Smart splitting function.
        """
        if isinstance(sent, str):
            return sent.split(" ")
        else:
            return sent

    def score(self, sentence, candidate):
        ensemble_inputs = [component.get_tokens(sentence) for component in self.ensembles]
        candidate_tokens = self._split(candidate)
        scoring_tokens = []
        for w in candidate_tokens:
            if w in self.target_vocab_map:
                scoring_tokens.append(self.target_vocab_map[w])
            else:
                scoring_tokens.append(self.target_vocab_map["UNK"])
        # scoring_tokens.append(self.config.target_vocab_size)
        _, score = self.beam_search(ensemble_inputs, scoring_tokens=scoring_tokens)
        return score

    def batch_score(self, input_path, output_path):
        """
        Score given translation results.
        :param input_path: translation results
        :param output_path: scoring output file
        """
        scoring_results = map(str.strip, open(input_path).readlines())

        # iterations
        n_total_lines = len(self.ensembles[0].inputs)
        fresult = open(output_path, "w")
        for i in range(n_total_lines):
            # get inputs
            ensemble_inputs = [component.input_tokens[i] for component in self.ensembles]
            # get scoring input tokens
            scoring_result = scoring_results[i]
            scoring_words = list(scoring_result) if self.config.char_based  else self._split(scoring_result)
            scoring_tokens = []
            for w in scoring_words:
                if w in self.target_vocab_map:
                    scoring_tokens.append(self.target_vocab_map[w])
                else:
                    scoring_tokens.append(self.target_vocab_map["UNK"])
            scoring_tokens.append(self.config.target_vocab_size)
            # score
            _, score = self.beam_search(ensemble_inputs, scoring_tokens=scoring_tokens, beam_size=1)
            fresult.write("%f\n" % score)
        fresult.close()

    def batch_translate(self, output_path, beam_size=20, include_score=False):
        """
        Translate.
        :param beam_size:
        :param scoring: flag of outputting re-ranking score
        :param beam_size: beam size of the decoding
        :param include_score: if including score in the result
        """
        # translation iterations
        total_bleu = 0.
        total_smoothed_bleu = 0.
        total_count = 0
        fresult = open(output_path, "w")

        # iterations
        n_total_lines = len(self.ensembles[0].inputs)
        for i in range(n_total_lines):
            print ("[S%d]" % (total_count + 1), self.ensembles[0].inputs[i])
            ensemble_inputs = [component.input_tokens[i] for component in self.ensembles]
            result, score = self.beam_search(ensemble_inputs, beam_size=beam_size)
            # Special case
            if not result:
                logging.info("search with beam size 100")
                result, score = self.beam_search(ensemble_inputs, beam_size=100)
            result_words = self._postprocess(self._split(self.ensembles[0].inputs[i]), result)
            if not result_words:
                result_words.append("EMPTY")
            print ("[T:%.2f]" % score, " ".join(result_words))
            output_line = " ".join(result_words) + "\n"
            if include_score:
                output_line = "%f ||| %s" % (score, output_line)
            fresult.write(output_line)
            total_count += 1
        # - #
        fresult.close()
        print ("Total count:", total_count)
        print ("Mean BLEU: %.2f" % (total_bleu / total_count))
        print ("Mean smoothed BLEU: %.2f" % (total_smoothed_bleu / total_count))


    def translate(self, sentence, beam_size=20):
        """
        Translate one sentence.
        :return: result, score
        """
        ensemble_inputs = [component.get_tokens(sentence) for component in self.ensembles]
        result, score = self.beam_search(ensemble_inputs, beam_size=beam_size)
        # Special case
        if result:
            result_words = self._postprocess(self._split(sentence), result)
            if not result_words:
                result_words.append("EMPTY")
            output_line = " ".join(result_words)
            return output_line, score
        else:
            return None, None

    def translate_nbest(self, sentence, beam_size=20, nbest=20):
        ensemble_inputs = [component.get_tokens(sentence) for component in self.ensembles]
        hyps = self.beam_search(ensemble_inputs, beam_size=beam_size, nbest=nbest)
        # Special case
        results = []
        for result, score in hyps:
            result_words = self._postprocess(self._split(sentence), result)
            if not result_words:
                result_words.append("EMPTY")
            output_line = " ".join(result_words)
            results.append((output_line, score))
        return results

    def _prepare(self):
        self.ensembles = [NeuralMTComponent(p, self.config) for p in self.config.paths()]
        logging.info("%d ensembles loaded" % len(self.ensembles))

        # vocabulary building
        self.target_vocab = pickle.load(open(self.config.target_vocab))
        self.target_vocab_map = {}
        for i, word in enumerate(self.target_vocab):
            self.target_vocab_map[word] = i

    # Old beam search code: https://gist.github.com/zomux/d49fd7c21ea46891cce0a60affe4ac88
    def beam_search(self, ensemble_inputs, beam_size=20, scoring_tokens=None, nbest=0):
        eol_token = self.target_vocab_map[self.config.end_token]
        sol_token = self.target_vocab_map[self.config.start_token]
        ensemble_weights = [p.weight for p in self.config.paths()]
        ensemble_weights = np.array(ensemble_weights) / np.sum(ensemble_weights)
        if scoring_tokens:
            max_len = len(scoring_tokens)
        else:
            max_len = min(max(10, int(ensemble_inputs[0].shape[0] * 3)), 100)
        ensemble_count = len(ensemble_inputs)
        ensemble_range = range(ensemble_count)
        ensemble_encoder_outputs = [self.ensembles[i].encoder.compute([ensemble_inputs[i]]) for i in ensemble_range]
        # Pick the first one in the batch
        for encoder_outputs in ensemble_encoder_outputs:
            for k in encoder_outputs:
                encoder_outputs[k] = encoder_outputs[k][0]

        # hyp: tokens, sum of -log
        final_hyps = []
        # hyp: state, tokens, sum of -log
        hyps = [{
            "states": [np.zeros((self.ensembles[i].hidden_size,), dtype="float32") for i in ensemble_range],
            "tokens": [sol_token],
            "logp": 0.
        }]

        for t in range(max_len):
            # state, tokens, new_token, sum of -log
            new_hyps = []
            # Run in batch mode
            batch_state_list = [[] for _ in ensemble_range]
            batch_last_token = []
            for hyp in hyps:
                for i in ensemble_range:
                    batch_state_list[i].append(hyp["states"][i])
                batch_last_token.append(hyp["tokens"][-1])
            batch_new_state_list = []
            for i in ensemble_range:
                decoder_inputs = [t, batch_state_list[i], batch_last_token] + [p[1] for p in sorted(ensemble_encoder_outputs[i].items())]
                batch_new_state_list.append(self.ensembles[i].decoder.compute(*decoder_inputs))
            batch_logprobs_list = [- np.log(self.ensembles[i].expander.compute(batch_new_state_list[i])) for i in ensemble_range]
            mean_batch_logprobs = sum(
                [batch_logprobs_list[i] * ensemble_weights[i] for i in range(len(batch_logprobs_list))])

            # Sort hyps
            for i, hyp in enumerate(hyps):
                new_states = [batch_new_state[i] for batch_new_state in batch_new_state_list]
                logprobs = mean_batch_logprobs[i] + hyp["logp"]
                if scoring_tokens:
                    best_indices = [scoring_tokens[t]]
                else:
                    best_indices = sorted(np.argpartition(logprobs, beam_size)[:beam_size], key=lambda x: logprobs[x])
                for idx in best_indices:
                    new_hyps.append({
                        "states": new_states,
                        "tokens": hyp["tokens"] + [idx],
                        "logp": logprobs[idx]
                    })
            new_hyps.sort(key=lambda h: h["logp"])
            # Get final hyps
            for i in range(len(new_hyps)):
                hyp = new_hyps[i]
                if hyp["tokens"][-1] == eol_token:
                    tokens = hyp["tokens"][1:-1]
                    final_hyps.append({
                        "tokens": tokens,
                        "logp": hyp["logp"] / len(tokens)
                    })
            # Update hyps
            hyps = [h for h in new_hyps if h["tokens"][-1] != eol_token][:beam_size]
            if not hyps:
                break
        # Sort final_hyps
        if nbest > 0:
            if not final_hyps:
                return []
            else:
                final_hyps.sort(key=lambda h: h["logp"])
                return final_hyps[:nbest]
        else:
            if not final_hyps:
                result = []
                score = 999
            else:
                final_hyps.sort(key=lambda h: h["logp"])
                result = final_hyps[0]["tokens"]
                score = final_hyps[0]["logp"]
            return result, score

    def _postprocess(self, src_words, result, mark_unk=False):
        result_words = []
        src_words = [w for w in src_words if w != "<s>"]
        in_unk_chunk = False
        for i, token in enumerate(result):
            word = self.target_vocab[token]
            if word.startswith("UNKPOS"):
                align_index = i + int(word.replace("UNKPOS", ""))
                if align_index >= 0 and align_index < len(src_words):
                    src_word = src_words[align_index]
                    if not in_unk_chunk:
                        in_unk_chunk = True
                        if align_index == 0:
                            src_word = "{%s" % src_word
                        else:
                            result_words.append("{%s" % src_words[align_index - 1])
                    result_words.append(src_word)
            else:
                if in_unk_chunk:
                    last_word = self.target_vocab[result[i - 1]]
                    last_align = i - 1 + int(last_word.replace("UNKPOS", ""))
                    next_align = last_align + 1
                    if next_align >= 0 and next_align < len(src_words):
                        result_words.append(src_words[next_align])
                    result_words[-1] += "}"
                    in_unk_chunk = False
                result_words.append(word)
        if in_unk_chunk:
            result_words[-1] += "}"
        return result_words



