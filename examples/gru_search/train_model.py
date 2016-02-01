#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging

logging.basicConfig(level=logging.INFO)
from argparse import ArgumentParser

from deepy import *
from deepy.trainers.trainers import FineTuningAdaGradTrainer
from neuralmt import TMCostLayer, SoftAttentionalLayer

theano.config.compute_test_value = 'warn'

counter = 0

def training_monitor():
    global trainer, data, valid_path, counter
    counter += 1
    if counter % 1500 == 0:
        trainer._run_valid(-1, data.valid_set(), dry_run=True)
    else:
        sys.stdout.write(" ITER COST:%.2f" % trainer.last_score)
        sys.stdout.flush()

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--model_path", default="/tmp/nmt_model.uncompressed.npz")
    ap.add_argument("--word_embed", default=500, type=int)
    ap.add_argument("--src_vocab_size", default=200000, type=int)
    ap.add_argument("--tgt_vocab_size", default=40000, type=int)
    ap.add_argument("--train_size", default=23442, type=int)
    ap.add_argument("--hidden_size", default=1000, type=int)
    ap.add_argument("--optimizer", default="adam")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip_iters", default=0, type=int)
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

    recurrent_unit = GRU(args.hidden_size, input_type="sequence", output_type="sequence", additional_input_dims=[args.word_embed])

    expander_chain = Chain(Dense(600), Dense(args.tgt_vocab_size))
    expander_chain.initialize(args.hidden_size)

    attention_layer = SoftAttentionalLayer(recurrent_unit)
    attention_var = attention_layer.belongs_to(decoder).compute(hidden_layer, mask=src_mask_var, feedback=second_input, steps=tgt_var.shape(1))

    # expander
    output_var = expander_chain.belongs_to(expander).compute(attention_var)

    cost = TMCostLayer(tgt_var, tgt_mask_var, args.tgt_vocab_size).compute(output_var)


    model = BasicNetwork(input_dim=[src_var, src_mask_var])
    model.training_callbacks.append(training_monitor)

    data = OnDiskDataset("/tmp/data_prefix_train.pkl",
                         valid_path="/tmp/data_prefix_valid.pkl",
                         train_size=args.train_size, cached=True, shuffle_memory=True)
    valid_set = data.valid_set()
    train_set = data.train_set()

    # Train
    training_config = {"learning_rate": shared_scalar(0.05),
                       "gradient_clipping": 3,
                       "auto_save": args.model_path,
                       "patience": 20}


    if args.optimizer == "adam":
        training_config["learning_rate"] = shared_scalar(0.001)
        training_config["patience"] = 10
        trainer = AdamTrainer(model, training_config)

        if args.resume:
            trainer.load_params(training_config["auto_save"])

        trainer.run(train_set, valid_set=valid_set, train_size=args.train_size,
                    controllers=[ScheduledLearningRateAnnealer(trainer,
                                                               start_halving_at=5 - args.skip_iters, end_at=8 - args.skip_iters)])
    else:
        if args.optimizer == "sgd":
            trainer = SGDTrainer(model, training_config)
        elif args.optimizer == "adadelta":
            trainer = AdaDeltaTrainer(model, training_config)
        elif args.optimizer == "momentum":
            trainer = MomentumTrainer(model, training_config)
        elif args.optimizer == "finetune":
            trainer = FineTuningAdaGradTrainer(model, training_config)

        trainer.run(train_set, valid_set=valid_set, train_size=args.train_size,
                    controllers=[ScheduledLearningRateAnnealer(trainer, start_halving_at=5, end_at=7)])
