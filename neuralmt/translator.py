#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *
from . import *
from config import NeuralMTConfiguration, NeuralMTPath

import pickle, gzip
from logging import Logger
logging = Logger(__name__)

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
        self.encoder, self.decoder, self.expander = self._get_neural_components(self.mt_path)
        self.vocab = pickle.load(open(self.mt_path.vocab))
        self.vocab_map = {}
        for i, word in enumerate(self.vocab):
            self.vocab_map[word] = i
        self.inputs = map(str.strip, open(self.mt_path.input_path).readlines())
        self.input_tokens = map(self.get_tokens, self.inputs)

    def get_tokens(self, line):
        if self.config.char_based:
            src_words = list(line)
        else:
            src_words = line.split(" ")
        src_words.insert(0, "<s>")
        src_tokens = []
        for w in src_words:
            if w in self.vocab_map:
                src_tokens.append(self.vocab_map[w])
            else:
                src_tokens.append(self.vocab_map["UNK"])
        src_tokens.reverse()
        return np.array(src_tokens)

    def _get_neural_components(self, mt_path):
        """
        Get neural encoder, decoder and expander.
        :type mt_path: NeuralMTPath
        """

        logging.info("loading model: %s" % mt_path.model)
        if mt_path.model.endswith(".gz"):
            saved_params = pickle.load(gzip.open(mt_path.model))
        else:
            saved_params = np.load(mt_path.model)


        # Create expander
        expander_model = BasicNetwork(input_dim=self.config.hidden_size, input_tensor=2)
        if self.config.approx_size:
            expander_model.stack(Dense(self.config.approx_size, 'linear'))
        expander_model.stack(Dense(self.config.target_vocab_size),
                             Softmax(),
                             LogProbLayer())

        # Create encoder and decoder
        if self.config.arch == "lstm_one_layer_search":
            # Encoder
            encoder_x = T.imatrix('x')
            input_mask = T.neq(encoder_x, -1)
            encoder_model = BasicNetwork(input_dim=mt_path.vocab_size, input_tensor=encoder_x)
            encoder_model.stack(WordEmbedding(self.config.word_embed, mt_path.vocab_size, zero_index=-1))
            forward_rnn = LSTM(self.config.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask)
            backward_rnn = LSTM(self.config.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask,
                                go_backwards=True)
            encoder_model.stack(Concatenate(forward_rnn,
                                Chain().stack(
                                  backward_rnn,
                                  Reverse3D()
                                )))
            # Decoder
            decoder_model = BasicNetwork(input_dim=self.config.hidden_size, input_tensor=T.ivector())

            seq_input = T.matrix('seq')
            decoder_model.input_variables.append(seq_input)
            state_input = T.matrix("s")
            cell_input = T.matrix("c")
            decoder_x = T.ivector('x')
            decoder_model.input_variables.append(state_input)
            decoder_model.input_variables.append(cell_input)
            decoder_model.input_variables.append(decoder_x)

            target_embed = Chain(0).stack(WordEmbedding(self.config.word_embed, self.config.target_vocab_size))
            decoder_model.register_layer(target_embed)
            decoder_input = target_embed.output(decoder_model.input_tensor)

            recurrent_unit =  LSTM(self.config.hidden_size, input_type="sequence", output_type="sequence", second_input_size=self.config.word_embed)
            decoder_core = TMSearchLayer(recurrent_unit, mask=T.neq(decoder_x, -1), predict_input=True, test=True)
            decoder_core.connect(self.config.hidden_size * 2)
            decoder_model.register_layer(decoder_core)
            # init_states = recurrent_unit.produce_initial_states(x)
            UaH = T.dot(seq_input, decoder_core.Ua)
            new_state, new_cell = decoder_core.step(decoder_input, state_input, cell_input, seq_input, UaH)
            decoder_model.stack(ResetLayer(T.concatenate([new_state, new_cell], axis=1)))

            if type(saved_params) == list:
                saved_params = saved_params[:len(saved_params)/3]
            else:
                saved_params = [saved_params["arr_%d" % k] for k in range(47)]

            _ = saved_params.pop(1)
            saved_params.insert(25, _)
        else:
            encoder_model, decoder_model = None
            raise SystemError("the arch '%s' is not supported." % self.config.arch)

        # Load parameters
        for param, saved_param in zip(encoder_model.parameters + decoder_model.parameters + expander_model.parameters,
                                     saved_params):

            param.set_value(saved_param)

        return encoder_model, decoder_model, expander_model


class NeuralTranslator(object):

    def __init__(self, config):
        """
        :type config: NeuralMTConfiguration
        """
        self.config = config
        self._prepare()

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
            scoring_words = list(scoring_result) if self.config.char_based  else scoring_result.split(" ")
            scoring_tokens = []
            for w in scoring_words:
                if w in self.target_vocab_map:
                    scoring_tokens.append(self.target_vocab_map[w])
                else:
                    scoring_tokens.append(self.target_vocab_map["UNK"])
            scoring_tokens.append(self.config.target_vocab_size)
            # score
            _, score = self._translate_core(ensemble_inputs, scoring_input=scoring_tokens, beam_size=1)
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
            print "[S%d]" % (total_count + 1), self.ensembles[0].inputs[i]
            ensemble_inputs = [component.input_tokens[i] for component in self.ensembles]
            result, score = self._translate_core(ensemble_inputs, beam_size=beam_size)
            # Special case
            if not result:
                logging.info("search with beam size 100")
                result, score = self._translate_core(ensemble_inputs, beam_size=100)
            result_words = self._postprocess(self.ensembles[0].inputs[i].split(" "), result)
            if not result_words:
                result_words.append("EMPTY")
            print "[T:%.2f]" % score, " ".join(result_words)
            output_line = " ".join(result_words) + "\n"
            if include_score:
                output_line = "%f ||| %s" % (score, output_line)
            fresult.write(output_line)
            total_count += 1
        # - #
        fresult.close()
        print "Total count:", total_count
        print "Mean BLEU: %.2f" % (total_bleu / total_count)
        print "Mean smoothed BLEU: %.2f" % (total_smoothed_bleu / total_count)

    def translate(self, sentence, beam_size=20):
        """
        Translate one sentence.
        :return: result, score
        """
        ensemble_inputs = [component.get_tokens(sentence) for component in self.ensembles]
        result, score = self._translate_core(ensemble_inputs, beam_size=beam_size)
        # Special case
        if result:
            result_words = self._postprocess(sentence.split(" "), result)
            if not result_words:
                result_words.append("EMPTY")
            output_line = " ".join(result_words)
            return output_line, score
        else:
            return None, None

    def _prepare(self):
        self.ensembles = [NeuralMTComponent(p, self.config) for p in self.config.paths()]
        logging.info("%d ensembles loaded" % len(self.ensembles))

        # vocabulary building
        self.target_vocab = pickle.load(open(self.config.target_vocab))
        self.target_vocab_map = {}
        for i, word in enumerate(self.target_vocab):
            self.target_vocab_map[word] = i

    def _translate_core(self, ensemble_inputs, beam_size=20, scoring_input=None):
        hidden_size = self.config.hidden_size
        eol_token = self.config.target_vocab_size
        ensemble_weights = [p.weight for p in self.config.paths()]
        ensemble_weights = np.array(ensemble_weights) / np.sum(ensemble_weights)
        if scoring_input:
            max_len = len(scoring_input)
        else:
            max_len = int(ensemble_inputs[0].shape[0] * 1.5)
        ensemble_count = len(ensemble_inputs)
        ensemble_range = range(ensemble_count)
        reps = [self.ensembles[i].encoder.compute([ensemble_inputs[i]])[0] for i in range(len(ensemble_inputs))]
        # hyp: tokens, sum of -log
        final_hyps = []
        # hyp: state, tokens, sum of -log
        hyps = [([np.zeros((hidden_size,), dtype="float32") for _ in ensemble_range], [np.zeros((hidden_size,), dtype="float32") for _ in ensemble_range],
                [0], 0.)]

        for time in range(max_len):
            time_beam = beam_size if time > 1 else beam_size * 3
            # state, tokens, new_token, sum of -log
            new_hyps = []
            # Run in batch mode
            batch_state_list = [[] for _ in ensemble_range]
            batch_cell_list = [[] for _ in ensemble_range]
            batch_last_token = []
            for states, cells, tokens, _ in hyps:
                for i in ensemble_range:
                    batch_state_list[i].append(states[i])
                    batch_cell_list[i].append(cells[i])
                batch_last_token.append(tokens[-1])
            batch_mixed_state_list = [self.ensembles[i].decoder.compute(batch_last_token, reps[i], batch_state_list[i], batch_cell_list[i], ensemble_inputs[i]) for i in ensemble_range]
            batch_new_state_list = [batch_mixed_state_list[i][:, :hidden_size] for i in ensemble_range]
            batch_new_cell_list = [batch_mixed_state_list[i][:, hidden_size:] for i in ensemble_range]
            batch_logprobs_list = [self.ensembles[i].expander.compute(batch_new_state_list[i]) for i in ensemble_range]

            mean_batch_logprobs = sum([batch_logprobs_list[i] * ensemble_weights[i] for i in range(len(batch_logprobs_list))])

            # Sort hyps
            for i in range(len(hyps)):
                states, cells, tokens, prev_logprob = hyps[i]
                new_states = [batch_new_state[i] for batch_new_state in batch_new_state_list]
                new_cells = [batch_new_cell[i] for batch_new_cell in batch_new_cell_list]
                logprobs = mean_batch_logprobs[i] + prev_logprob
                if scoring_input:
                    best_indices = [scoring_input[time]]
                else:
                    if time_beam > len(logprobs):
                        time_beam = int(len(logprobs) * 0.8)
                    best_indices = sorted(np.argpartition(logprobs, time_beam)[:time_beam], key=lambda x: logprobs[x])
                for idx in best_indices:
                    new_hyps.append((new_states, new_cells, tokens, idx, logprobs[idx]))
            new_hyps.sort(key=lambda h: h[-1])
            # Get final hyps
            for i in range(min(time_beam * 2, len(new_hyps))):
                hyp = new_hyps[i]
                if hyp[3] == eol_token:
                    tokens = hyp[2][1:]
                    final_hyps.append((tokens, hyp[4] / len(tokens)))
            # Save to hyps
            hyps = [(h[0], h[1], h[2] + [h[3]], h[4]) for h in new_hyps if h[3] != eol_token][:time_beam]
        # Sort final_hyps
        if not final_hyps:
            result = []
            score = 999
        else:
            final_hyps.sort(key=lambda h: h[1])
            result = final_hyps[0][0]
            score = final_hyps[0][1]
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



