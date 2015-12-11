#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)
from argparse import ArgumentParser
import pickle

from deepy import *
from deepy.trainers.trainers import FineTuningAdaGradTrainer
from neuralmt import NeuralTM, TMCostLayer, SoftAttentionalLayer

default_model = "/tmp/default_model.gz"
counter = 0

def training_monitor():
    global trainer, data, valid_path, counter
    counter += 1
    if counter % 1500 == 0:
        trainer._run_valid(-1, data.valid_set())
    else:
        sys.stdout.write(" ITER COST:%.2f" % trainer.last_score)
        sys.stdout.flush()

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--valid", default=None, type=str)
    ap.add_argument("--save", default=default_model, type=str)
    ap.add_argument("--load", default=None, type=str)
    ap.add_argument("--hidden_size", default=512, type=int)
    ap.add_argument("--source_size", default=10000, type=int)
    ap.add_argument("--target_size", default=10001, type=int)
    ap.add_argument("--train_size", default=(1000000 / 128), type=int)
    ap.add_argument("--arch", default="one_layer")
    ap.add_argument("--iter_offset", default=0, type=int)
    ap.add_argument("--approx", action="store_true")
    ap.add_argument("--target_vocab", default="", type=str)
    ap.add_argument("--lr", default=0.1, type=float)
    ap.add_argument("--gradient_clip", default=3, type=float)
    ap.add_argument("--word_embed", default=0, type=int)
    ap.add_argument("--skip", default=0, type=int)
    ap.add_argument("--encoder_mask", action="store_true")
    ap.add_argument("--optimizer", default="sgd", type=str)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--shuffle_memory", action="store_true")
    ap.add_argument("--better_lstm", action="store_true")
    ap.add_argument("--awful_lstm", action="store_true")
    ap.add_argument("--fresh_load", action="store_true")
    ap.add_argument("--freeze_embed", action="store_true")
    args = ap.parse_args()

    input_var = T.imatrix("x")
    if args.encoder_mask:
        input_mask = T.neq(input_var, -1)
    else:
        input_mask = None

    if args.awful_lstm:
        encoder_bf = 1
        decoder_bf = 1
    elif args.better_lstm:
        encoder_bf = 0.3
        decoder_bf = 1
    else:
        encoder_bf = decoder_bf = 0

    model = NeuralTM(input_dim=args.source_size, input_tensor=input_var,
                     monitor_callback=training_monitor)
    # Embedding
    if args.word_embed:
        encoder_embed = WordEmbedding(args.word_embed, args.source_size, zero_index=-1)
        model.stack(encoder_embed)
    else:
        model.stack(OneHotEmbedding(args.source_size, cached=False, zero_index=-1))

    # Target embedding
    if args.predict:
        # TODO: refactor this stuff
        decoder_embed = WordEmbedding(args.word_embed, args.target_size, zero_index=0)
        target_embed = Chain(0).stack(decoder_embed)
        model.register_layer(target_embed)
        target_matrix = T.concatenate([T.zeros((model.target_matrix.shape[0], 1), dtype="int32"), model.target_matrix[:, :-1]], axis=1)
        second_input = target_embed.output(target_matrix).dimshuffle(1, 0, 2)
        second_input_size = args.word_embed
    else:
        second_input = second_input_size = None

    if args.arch == "one_layer":

        model.stack(# Encoder
                    IRNN(args.hidden_size, input_type="sequence", output_type="one",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask),
                    # Decoder
                    IRNN(args.hidden_size, input_type="one", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9,
                         steps=model.target_length))

    elif args.arch == "lstm_one_layer":

        model.stack(# Encoder
                    LSTM(args.hidden_size, input_type="sequence", output_type="one", mask=input_mask, forget_bias=decoder_bf),
                    # Decoder
                    LSTM(args.hidden_size, input_type="one", output_type="sequence",
                         steps=model.target_length, forget_bias=decoder_bf))

    elif args.arch == "two_layer":

        model.stack(# Encoder
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask, forget_bias=encoder_bf),
                    IRNN(args.hidden_size, input_type="sequence", output_type="one",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask, forget_bias=encoder_bf),
                    # Decoder
                    IRNN(args.hidden_size, input_type="one", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9,
                         steps=model.target_length,
                         second_input=second_input, second_input_size=second_input_size, forget_bias=decoder_bf),
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9, forget_bias=decoder_bf,
                         steps=model.target_length))

    elif args.arch == "three_layer":

        model.stack(# Encoder
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask),
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask),
                    IRNN(args.hidden_size, input_type="sequence", output_type="one",
                         bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask),
                    # Decoder
                    IRNN(args.hidden_size, input_type="one", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.99,
                         steps=model.target_length,
                         second_input=second_input, second_input_size=second_input_size),
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.99,
                         steps=model.target_length),
                    IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                         bound_recurrent_weight=False, weight_scale=0.99,
                         steps=model.target_length))

    elif args.arch == "lstm_two_layer":

        model.stack(# Encoder
                    LSTM(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask, forget_bias=encoder_bf),
                    LSTM(args.hidden_size, input_type="sequence", output_type="one", mask=input_mask, forget_bias=encoder_bf),
                    # Decoder
                    LSTM(args.hidden_size, input_type="one", output_type="sequence",
                         steps=model.target_length,
                         second_input=second_input, second_input_size=second_input_size, forget_bias=decoder_bf),
                    LSTM(args.hidden_size, input_type="sequence", output_type="sequence",
                         steps=model.target_length, forget_bias=decoder_bf))

    elif args.arch == "lstm_transform_two_layer":

        model.stack(LSTM(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask, forget_bias=encoder_bf),
                    LSTM(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask, forget_bias=decoder_bf))

    elif args.arch == "one_layer_search":
        # Encoder
        forward_rnn = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
             bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask)
        backward_rnn = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
             bound_recurrent_weight=False, weight_scale=0.9, mask=input_mask, go_backwards=True)
        model.stack(Concatenate(forward_rnn,
                                Chain().stack(
                                    backward_rnn,
                                    Reverse3D()
                                )))
        # Decoder
        recurrent_unit = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                              bound_recurrent_weight=False, weight_scale=0.9, second_input_size=second_input_size)
        decoder = SoftAttentionalLayer(recurrent_unit, steps=model.target_length, mask=input_mask, predict_input=second_input)
        model.stack(decoder)
    elif args.arch == "tuned_one_layer_search":
        # Encoder
        forward_rnn = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
             bound_recurrent_weight=False, weight_scale=0.3, mask=input_mask)
        backward_rnn = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
             bound_recurrent_weight=False, weight_scale=0.3, mask=input_mask, go_backwards=True)
        model.stack(Concatenate(forward_rnn,
                                Chain().stack(
                                    backward_rnn,
                                    Reverse3D()
                                )))
        # Decoder
        recurrent_unit = IRNN(args.hidden_size, input_type="sequence", output_type="sequence",
                              bound_recurrent_weight=False, weight_scale=0.8, second_input_size=second_input_size)
        decoder = SoftAttentionalLayer(recurrent_unit, steps=model.target_length, mask=input_mask, predict_input=second_input)
        model.stack(decoder)

    elif args.arch == "lstm_one_layer_search":
        # Encoder
        forward_rnn = LSTM(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask, forget_bias=encoder_bf)
        backward_rnn = LSTM(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask,
                            go_backwards=True, forget_bias=decoder_bf)
        model.stack(Concatenate(forward_rnn,
                                Chain().stack(
                                    backward_rnn,
                                    Reverse3D()
                                )))
        # Decoder
        recurrent_unit = LSTM(args.hidden_size, input_type="sequence", output_type="sequence", second_input_size=second_input_size)
        decoder = SoftAttentionalLayer(recurrent_unit, steps=model.target_length, mask=input_mask, predict_input=second_input)
        model.stack(decoder)
    elif args.arch == "gru_one_layer_search":
        # Encoder
        forward_rnn = GRU(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask)
        backward_rnn = GRU(args.hidden_size, input_type="sequence", output_type="sequence", mask=input_mask,
                            go_backwards=True)
        model.stack(Concatenate(forward_rnn,
                                Chain().stack(
                                    backward_rnn,
                                    Reverse3D()
                                )))
        # Decoder
        recurrent_unit = GRU(args.hidden_size, input_type="sequence", output_type="sequence", second_input_size=second_input_size)
        decoder = SoftAttentionalLayer(recurrent_unit, steps=model.target_length, mask=input_mask, predict_input=second_input)
        model.stack(decoder)

    if args.approx:
        model.stack(# A small layer before full output
                    Dense(600, 'linear'))

    # Output layer
    model.stack(Dense(args.target_size))

    model.stack(TMCostLayer(model.target_matrix, model.target_mask, args.target_size))

    data = OnDiskDataset(args.data, valid_path=args.valid, train_size=args.train_size,
                         cached=True, shuffle_memory=args.shuffle_memory)
    valid_set = data.valid_set()
    train_set = data.train_set()

    # Train
    training_config = {"learning_rate": LearningRateAnnealer.learning_rate(args.lr),
                       "gradient_clipping": args.gradient_clip if args.gradient_clip > 0 else None,
                       "auto_save": args.save,
                       "patience": 20}
    if args.freeze_embed:
        training_config["freeze_params"] = [encoder_embed.embed_matrix, decoder_embed.embed_matrix]

    if args.optimizer == "sgd":
        trainer = SGDTrainer(model, training_config)
    elif args.optimizer == "adam":
        trainer = AdamTrainer(model, training_config)
    elif args.optimizer == "adadelta":
        trainer = AdaDeltaTrainer(model, training_config)
    elif args.optimizer == "momentum":
        trainer = MomentumTrainer(model, training_config)
    elif args.optimizer == "finetune":
        trainer = FineTuningAdaGradTrainer(model, training_config)

    if args.load:
            trainer.load_params(args.load, exclude_free_params=args.fresh_load)
    if args.profile:
        valid_set = None

    if args.skip:
        trainer.skip(args.skip)

    trainer.run(train_set, valid_set=valid_set, train_size=args.train_size,
        controllers=[ScheduledLearningRateAnnealer(trainer, iter_start_halving=5 - args.iter_offset, max_iters=10 - args.iter_offset)])

    if not args.profile:
        model.save_params(args.save)
