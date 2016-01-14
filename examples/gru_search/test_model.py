
import logging

logging.basicConfig(level=logging.INFO)
from argparse import ArgumentParser

from deepy import *
from neuralmt import SoftAttentionalLayer, LogProbLayer, NeuralMTConfiguration, NeuralTranslator

theano.config.compute_test_value = 'raise'


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model_path", default="/tmp/nmt_model.uncompressed.npz")
    ap.add_argument("--word_embed", default=500)
    ap.add_argument("--src_vocab_size", default=200000)
    ap.add_argument("--tgt_vocab_size", default=40000)
    ap.add_argument("--hidden_size", default=1000)
    ap.add_argument("--src_vocab", default="PATH TO ENGLISH VOCAB")
    ap.add_argument("--tgt_vocab", default="PATH TO GERMAN VOCAB")
    args = ap.parse_args()


    src_var = create_var(T.imatrix(), test_shape=[64, 10], test_dtype="int32")

    encoder = Block()


    # embedding
    encoder_embed = WordEmbedding(args.word_embed, args.src_vocab_size).link(encoder).compute(src_var)

    # encoder
    forward_rnn_var = (GRU(args.hidden_size,input_type="sequence", output_type="sequence")
                       .link(encoder).compute(encoder_embed))
    backward_rnn_var = Chain(GRU(args.hidden_size, input_type="sequence", output_type="sequence", backward=True),
                             Reverse3D()).link(encoder).compute(encoder_embed)
    encoder_output_var = Concatenate(axis=2).compute(forward_rnn_var, backward_rnn_var)

    # decoder
    decoder = Block()

    last_token_var = create_var(T.ivector("tok"), test_shape=[64], test_dtype="int32")
    seq_input_var = create_var(T.matrix('seq'), dim=args.hidden_size * 2, test_shape=[64, args.hidden_size * 2])
    state_var = create_var(T.matrix("s"), dim=args.hidden_size, test_shape=[64, args.hidden_size])

    input_embed = WordEmbedding(args.word_embed, args.tgt_vocab_size).link(decoder).compute(last_token_var)

    recurrent_unit = GRU(args.hidden_size, input_type="sequence", output_type="sequence", additional_input_dims=[input_embed.dim()])
    attention_layer = SoftAttentionalLayer(recurrent_unit, test=True)
    attention_layer.link(decoder).connect(args.hidden_size * 2)

    new_state = attention_layer.step({
        "UaH": T.dot(seq_input_var.tensor, attention_layer.Ua),
        "feedback": input_embed.tensor,
        "inputs": seq_input_var.tensor,
        "state": state_var.tensor
    })["state"]
    decoder_output_var = create_var(new_state, dim=args.hidden_size)

    # expander

    expander = Block()

    expander_input_var = create_var(T.matrix("expander_input"), dim=args.hidden_size, test_shape=[64, args.hidden_size])

    dense_var = Chain(Dense(600), Dense(args.tgt_vocab_size)).link(expander).compute(expander_input_var)

    expander_output_var = Chain(Softmax(), LogProbLayer()).compute(dense_var)

    ####

    encoder_network = BasicNetwork(input_vars=[src_var], blocks=[encoder], output=encoder_output_var)
    decoder_network = BasicNetwork(input_vars=[last_token_var, seq_input_var, state_var], blocks=[decoder], output=decoder_output_var)
    expander_network = BasicNetwork(input_vars=[expander_input_var], blocks=[expander], output=expander_output_var)

    fill_parameters(args.model_path, [encoder_network, decoder_network, expander_network])

    config = NeuralMTConfiguration(
            target_vocab=args.tgt_vocab,
            target_vocab_size=args.tgt_vocab_size,
            hidden_size=args.hidden_size
        ).add_path(
            None, args.src_vocab, args.src_vocab_size, encoder_network, decoder_network, expander_network
        )

    translator = NeuralTranslator(config)
    result, score = translator.translate("<s> this is a good system </s>", beam_size=5)
    print result
    print score