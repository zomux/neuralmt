#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
WMT_ROOT = os.environ["WMT_ROOT"]
assert WMT_ROOT

from argparse import ArgumentParser

from deepy import *
from deepy.trainers.trainers import FineTuningAdaGradTrainer
from neuralmt import TMCostLayer, SoftAttentionalLayer

theano.config.compute_test_value = 'ignore'

counter = 0

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model_path", default="{}/models/wmt15_de-en.uncompressed.npz".format(WMT_ROOT))
    ap.add_argument("--word_embed", default=600, type=int)
    ap.add_argument("--src_vocab_size", default=50000, type=int)
    ap.add_argument("--tgt_vocab_size", default=50000, type=int)
    ap.add_argument("--hidden_size", default=1000, type=int)
    args = ap.parse_args()

    src_var = create_var(T.imatrix(), test_shape=[64, 10], test_dtype="int32")
    src_mask_var = create_var(T.matrix(), test_shape=[64, 10], test_dtype="float32")
    tgt_var = create_var(T.imatrix(), test_shape=[64, 10], test_dtype="int32")
    tgt_mask_var = create_var(T.matrix(), test_shape=[64, 10], test_dtype="float32")

    encoder = Block()
    decoder = Block()
    expander = Block()

    src_embed_layer = WordEmbedding(args.word_embed, args.src_vocab_size)

    encoder_embed = src_embed_layer.belongs_to(encoder).compute(src_var, mask=src_mask_var)

    # encoder
    forward_rnn_var = (GRU(args.hidden_size,input_type="sequence", output_type="sequence", mask=src_mask_var)
                       .belongs_to(encoder).compute(encoder_embed))
    backward_rnn_var = Chain(GRU(args.hidden_size, input_type="sequence", output_type="sequence", mask=src_mask_var, backward=True),
                             Reverse3D()).belongs_to(encoder).compute(encoder_embed)
    hidden_layer = Concatenate(axis=2).compute(forward_rnn_var, backward_rnn_var)

    # decoder
    # the first token is <s>=1
    feedback_var = tgt_var.apply(lambda t: T.concatenate([T.ones((t.shape[0], 1), dtype="int32"), t[:, :-1]], axis=1))

    tgt_embed_layer = WordEmbedding(args.word_embed, args.tgt_vocab_size)
    tgt_embed_layer.initialize(1)

    second_input = tgt_embed_layer.belongs_to(decoder).compute(feedback_var, mask=tgt_mask_var)

    second_input = DimShuffle(1, 0, 2).compute(second_input)


    recurrent_unit = LSTM(args.hidden_size, input_type="sequence", output_type="sequence", additional_input_dims=[args.word_embed])

    attention_layer = SoftAttentionalLayer(recurrent_unit)
    attention_var = attention_layer.belongs_to(decoder).compute(hidden_layer, mask=src_mask_var, feedback=second_input, steps=tgt_var.shape[1])

    # expander
    output_var = Chain(Dense(600), Dense(args.tgt_vocab_size)).belongs_to(expander).compute(attention_var)

    cost = TMCostLayer(tgt_var, tgt_mask_var, args.tgt_vocab_size).compute(output_var)


    model = ComputationalGraph(input_vars=[src_var, src_mask_var],
                               target_vars=[tgt_var, tgt_mask_var],
                               blocks=[encoder, decoder, expander],
                               cost=cost)

    data = OnDiskDataset("{}/wmt15.de-en1_train.pkl".format(WMT_ROOT),
                         valid_path="{}/wmt15.de-en1_valid.pkl".format(WMT_ROOT),
                         cached=True, shuffle_memory=False)

    # Train
    training_config = {"gradient_clipping": 3,
                       "auto_save": args.model_path,
                       "patience": 20}

    trainer = MultiGPUTrainer(model, training_config, method='sgd',
                              learning_rate=1.0, step_len=20)

    trainer.run(data)