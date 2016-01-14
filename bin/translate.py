#!/usr/bin/env python
# -*- coding: utf-8 -*-

from deepy import *
import logging
logging.basicConfig(level=logging.INFO)
import os, sys
from argparse import ArgumentParser

from neuralmt import NeuralTranslator, NeuralMTConfiguration

if __name__ == '__main__':

    ap = ArgumentParser()
    ap.add_argument("--load", default=None, type=str)
    ap.add_argument("--model2", default=None, type=str)
    ap.add_argument("--model3", default=None, type=str)
    ap.add_argument("--model4", default=None, type=str)
    ap.add_argument("--model5", default=None, type=str)
    ap.add_argument("--hidden_size", default=1024, type=int)
    ap.add_argument("--source_size", default=80000, type=int)
    ap.add_argument("--target_size", default=40001, type=int)
    ap.add_argument("--arch", default="lstm_one_layer_search")
    ap.add_argument("--approx")
    ap.add_argument("--source_vocab", default="data/vocab_en_80k.pkl", type=str)
    ap.add_argument("--source_vocab2", default="data/vocab_normal_en_80k.pkl", type=str)
    ap.add_argument("--target_vocab", default="data/vocab_ja_40k.pkl", type=str)
    ap.add_argument("--word_embed", default=1000, type=int)
    ap.add_argument("--data", default="/home/ubuntu/data/pickles/remt1.v80k_40k.unkpos.b80.trun40.rev.valid.pack")
    ap.add_argument("--beam_size", default=5, type=int)
    ap.add_argument("--length_penalty", default="", type=str)
    ap.add_argument("--dump", default="/tmp/nmt_translate")
    ap.add_argument("--input", default="data/data.en", type=str)
    ap.add_argument("--input2", default="data/normal_order.en", type=str)
    ap.add_argument("--mark_unk", action="store_true")
    ap.add_argument("--weight", action="store_true")
    ap.add_argument("--character", action="store_true")
    ap.add_argument("--rerank", type=str)
    args = ap.parse_args()

    config = NeuralMTConfiguration(
            target_vocab=args.target_vocab,
            target_vocab_size=args.target_size,
            hidden_size=args.hidden_size,
            char_based=args.character
        ).add_path(
            args.input, args.load, args.source_vocab, args.source_size
        )

    translator = NeuralTranslator(config)
    translator.batch_translate(args.dump, args.beam_size, True)

