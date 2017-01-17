#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if "MT_ROOT" not in os.environ:
    raise Exception("Environment variable MT_ROOT is not found.")
MT_ROOT = os.environ["MT_ROOT"]
MODEL_ROOT = "{}/models".format(MT_ROOT)


from argparse import ArgumentParser
from deepy.dataset import OnDiskDataset
from deepy.trainers import ScheduledLearningRateAnnealer
from neuralmt import AttentionalNMT, NeuralVocab

import logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("--src_vocab", default="{}/comtrans_en.vocab".format(MODEL_ROOT))
    ap.add_argument("--tgt_vocab", default="{}/comtrans_de.vocab".format(MODEL_ROOT))
    ap.add_argument("--hidden_size", default=500, type=int)
    ap.add_argument("--embed_size", default=500, type=int)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()


    data = OnDiskDataset("{}/text/comtrans_train.pkl".format(MT_ROOT),
                         valid_path="{}/text/comtrans_valid.pkl".format(MT_ROOT),
                         cached=True, shuffle_memory=False)

    src_vocab_size = NeuralVocab(args.src_vocab).size()
    tgt_vocab_size = NeuralVocab(args.tgt_vocab).size()
    mt_model = AttentionalNMT(args.hidden_size, args.embed_size, src_vocab_size, tgt_vocab_size)


    if args.train:
        training_config = {"learning_rate": 0.0001,
                           "gradient_clipping": 3,
                           "patience": 20}

        trainer = mt_model.get_trainer(method='adam', config=training_config,
                                       valid_freq=300,
                                       annealer=ScheduledLearningRateAnnealer(7, 10),
                                       save_path=os.path.join(MODEL_ROOT, "comtrans_nmt1.npz"),
                                       valid_criteria='mixed')
        trainer.run(data)
    elif args.test:
        translator = mt_model.get_translator(args.src_vocab, args.tgt_vocab, os.path.join(MODEL_ROOT, "comtrans_nmt1.npz"))
        result, score = translator.translate("fire", beam_size=20)
        print result
        print score

    else:
        print ("You shall specify the action by --train or --test")