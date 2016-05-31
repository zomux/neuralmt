# coding: utf-8
import logging, sys

import os
WMT_ROOT = os.environ["WMT_ROOT"]
assert WMT_ROOT

logging.basicConfig(level=logging.INFO)
from argparse import ArgumentParser

from deepy import *
from neuralmt import SoftAttentionalLayer, LogProbLayer, NeuralMTConfiguration, NeuralTranslator

theano.config.compute_test_value = 'ignore'

model_name = "wmt15_de-en"
model_path = "{}/models/{}.uncompressed.npz".format(WMT_ROOT, model_name)
src_vocab = "{}/models/wmt15.de-en.de50k.vocab".format(WMT_ROOT)
tgt_vocab = "{}/models/wmt15.de-en.en50k.vocab".format(WMT_ROOT)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model_path", default=model_path)
    ap.add_argument("--word_embed", default=600)
    ap.add_argument("--src_vocab_size", default=50000)
    ap.add_argument("--tgt_vocab_size", default=50000)
    ap.add_argument("--hidden_size", default=1000)
    ap.add_argument("--src_vocab", default=src_vocab)
    ap.add_argument("--tgt_vocab", default=tgt_vocab)
    args = ap.parse_args()

    src_var = create_var(T.imatrix(), test_shape=[64, 10], test_dtype="int32")

    encoder = Block()

    # embedding
    encoder_embed = WordEmbedding(args.word_embed, args.src_vocab_size).belongs_to(encoder).compute(src_var)

    # encoder
    forward_rnn_var = (GRU(args.hidden_size, input_type="sequence", output_type="sequence")
                       .belongs_to(encoder).compute(encoder_embed))
    backward_rnn_var = Chain(GRU(args.hidden_size, input_type="sequence", output_type="sequence", backward=True),
                             Reverse3D()).belongs_to(encoder).compute(encoder_embed)
    encoder_output_var = Concatenate(axis=2).compute(forward_rnn_var, backward_rnn_var)

    # decoder
    decoder = Block()

    last_token_var = create_var(T.ivector("tok"), test_shape=[64], test_dtype="int32")
    seq_input_var = create_var(T.matrix('seq'), dim=args.hidden_size * 2, test_shape=[64, args.hidden_size * 2])
    state_var = create_var(T.matrix("s"), dim=args.hidden_size, test_shape=[64, args.hidden_size])

    input_embed = WordEmbedding(args.word_embed, args.tgt_vocab_size).belongs_to(decoder).compute(last_token_var)

    recurrent_unit = GRU(args.hidden_size, input_type="sequence", output_type="sequence",
                         additional_input_dims=[input_embed.dim()])
    attention_layer = SoftAttentionalLayer(recurrent_unit)
    attention_layer.belongs_to(decoder).initialize(input_dim=args.hidden_size * 2)

    step_parameters = attention_layer.get_step_inputs(seq_input_var, state=state_var, feedback=input_embed)

    new_state = attention_layer.step(step_parameters)["state"]
    new_state.output_dim = args.hidden_size
    decoder_output_var = new_state

    # expander

    expander = Block()

    expander_input_var = create_var(T.matrix("expander_input"), dim=args.hidden_size, test_shape=[64, args.hidden_size])

    dense_var = Chain(Dense(600), Dense(args.tgt_vocab_size)).belongs_to(expander).compute(expander_input_var)

    expander_output_var = Chain(Softmax(), LogProbLayer()).compute(dense_var)

    ####

    encoder_network = ComputationalGraph(input_vars=[src_var], blocks=[encoder], output=encoder_output_var)
    decoder_network = ComputationalGraph(input_vars=[last_token_var, seq_input_var, state_var], blocks=[decoder],
                                         output=decoder_output_var)
    expander_network = ComputationalGraph(input_vars=[expander_input_var], blocks=[expander],
                                          output=expander_output_var)

    fill_parameters(args.model_path, [encoder_network, decoder_network, expander_network])

    config = NeuralMTConfiguration(
        target_vocab=args.tgt_vocab,
        target_vocab_size=args.tgt_vocab_size,
        hidden_size=args.hidden_size
    ).add_path(
        None, args.src_vocab, args.src_vocab_size, encoder_network, decoder_network, expander_network
    )

    translator = NeuralTranslator(config)
    sentence = "<s> der Bau und die Reparatur der Autostraßen </s>"
    print "translating {}".format(sentence)
    result, _ = translator.translate(sentence, beam_size=5)
    print result

