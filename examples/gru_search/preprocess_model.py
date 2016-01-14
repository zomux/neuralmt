#!/usr/bin/env python
# -*- coding: utf-8 -*-

from stream import *
from neuralmt import SequentialDataBuilder

import logging
logging.basicConfig(level=logging.INFO)

BATCH_SIZE = 64

if __name__ == '__main__':
    logging.info("read raw data")
    en_data = open("PATH TO SOME DATA IN ENGLISH").readlines() >> map(str.strip) >> list
    ja_data = open("PATH TO SOME DATA IN GERMAN").readlines() >> map(str.strip) >> list
    paired_data = list(zip(en_data, ja_data))
    paired_data.sort(key=lambda p: p[1].count(" "))
    logging.info("%d lines of data" % len(paired_data))
    # Reduced data
    src_sent_pool = set()
    reduced_paired_data = []
    for en, ja in paired_data:
        if en.count(" ") < 4:
            continue
        if en not in src_sent_pool:
            reduced_paired_data.append((en, ja))
            src_sent_pool.add(en)
    logging.info("reduced to {} data points".format(len(reduced_paired_data)))
    paired_data = reduced_paired_data
    # Transform vocabulary
    logging.info("transform vocabulary")
    builder = SequentialDataBuilder()
    src_data = builder.transform("PATH TO THE ENGLISH VOCABULARY", paired_data >> map(itemgetter(0)) >> map(methodcaller("split", " ")),
                                 additional_head="<s>", additional_tail="</s>")
    tgt_data = builder.transform("PATH TO THE GERMAN VOCABULARY", paired_data >> map(itemgetter(1)) >> map(methodcaller("split", " ")),
                                 additional_tail="</s>")
    # Truncate
    logging.info("truncate")
    src_data, tgt_data = builder.truncate([src_data, tgt_data], source_len=50, target_len=60)
    logging.info("final datas = %d" % len(src_data))
    # Make batches
    logging.info("make batches")
    src_batches, src_mask = builder.make_batches(src_data, BATCH_SIZE, output_mask=True)
    tgt_batches, tgt_mask = builder.make_batches(tgt_data, BATCH_SIZE, output_mask=True)
    logging.info("%d batches in all" % len(src_batches))
    logging.info("dump data")
    builder.dump([src_batches, src_mask, tgt_batches, tgt_mask], "/tmp/data_prefix", valid_batches=50)