#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
if "MT_ROOT" not in os.environ:
    raise Exception("Environment variable MT_ROOT is not found.")
MT_ROOT = os.environ["MT_ROOT"]
MODEL_ROOT = "{}/models".format(MT_ROOT)

if not os.path.exists(MODEL_ROOT):
    os.mkdir(MODEL_ROOT)

from stream import *
from neuralmt import SequentialDataBuilder, NeuralVocab

import logging
logging.basicConfig(level=logging.INFO)

source_path = "{}/text/comtrans.en".format(MT_ROOT)
target_path = "{}/text/comtrans.de".format(MT_ROOT)

BATCH_SIZE = 32

if __name__ == '__main__':
    logging.info("making vocabulary")
    # Make vocab for de
    vocab_de = NeuralVocab()
    vocab_de.build(target_path, limit=50 * 1000)
    vocab_de.save(os.path.join(MODEL_ROOT, "comtrans_de.vocab"))
    # Make vocab for en
    vocab_en = NeuralVocab()
    vocab_en.build(source_path, limit=50 * 1000)
    vocab_en.save(os.path.join(MODEL_ROOT, "comtrans_en.vocab"))
    # Read data
    logging.info("read raw data")
    en_data = open(source_path).readlines() >> map(str.strip) >> list
    de_data = open(target_path).readlines() >> map(str.strip) >> list
    paired_data = list(zip(en_data, de_data))
    paired_data.sort(key=lambda p: p[1].count(" "))
    logging.info("%d lines of data" % len(paired_data))
    # Reduced data
    src_sent_pool = set()
    reduced_paired_data = []
    for en, de in paired_data:
        if not en or not de:
            continue
        en_len = float(en.count(" ") + 1)
        de_len = float(de.count(" ") + 1)
        if en_len / de_len > 2 or de_len / en_len > 2:
            continue
        if en_len > 50 or de_len > 50:
            continue
        if en not in src_sent_pool:
            reduced_paired_data.append((en, de))
            src_sent_pool.add(en)
    logging.info("reduced to {} data points".format(len(reduced_paired_data)))
    paired_data = reduced_paired_data
    # Transform vocabulary
    logging.info("transform vocabulary")
    builder = SequentialDataBuilder()
    src_data = builder.transform("{}/comtrans_en.vocab".format(MODEL_ROOT),
                                 paired_data >> map(itemgetter(0)) >> map(methodcaller("split", " ")))
    tgt_data = builder.transform("{}/comtrans_de.vocab".format(MODEL_ROOT),
                                 paired_data >> map(itemgetter(1)) >> map(methodcaller("split", " ")),
                                 additional_tail="</s>")
    # Truncate
    logging.info("truncate")
    src_data, tgt_data = builder.truncate([src_data, tgt_data], source_len=50, target_len=50)
    logging.info("after truncation, final data size = %d" % len(src_data))
    # Make batches
    logging.info("make batches")
    src_batches, src_mask = builder.make_batches(src_data, BATCH_SIZE, output_mask=True)
    tgt_batches, tgt_mask = builder.make_batches(tgt_data, BATCH_SIZE, output_mask=True)
    logging.info("%d batches in all" % len(src_batches))
    logging.info("dump data")
    builder.dump([src_batches, src_mask, tgt_batches, tgt_mask],
                 "{}/text/comtrans".format(MT_ROOT), valid_batches=50)