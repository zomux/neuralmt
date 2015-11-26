#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy_reg
import types

def _reduce_method(meth):
    return (getattr,(meth.__self__, meth.__func__.__name__))
copy_reg.pickle(types.MethodType, _reduce_method)

import os, sys
import numpy as np
import random
from argparse import ArgumentParser
from collections import Counter
import cPickle as pickle
from deepy import StreamPickler, onehot
from multiprocessing import Pool, cpu_count

class DataPacker(object):

    def __init__(self, source_size=10000, target_size=10000, batch_size=80,
                 sort_by_length=True, additional_target=True, additional_source=True,
                 avoid_valid_data=False, truncate_length=None, fix_length=False,
                 reverse_source=True):
        self.source_size = source_size
        self.target_size = target_size
        self.batch_size = batch_size
        self.sort_by_length = sort_by_length
        self.additional_target = additional_target
        self.additional_source = additional_source
        self.avoid_valid_data = avoid_valid_data
        self.truncate_length = truncate_length
        self.fix_length = fix_length
        self.reverse_source = reverse_source
        if additional_target:
            # Add EOL to the end of targets
            self.target_size += 1

    def process(self, data_fp, output_fp, pick=None):
        # # Sort
        out = open(output_fp, "wb")
        datas = []
        max_target_len = self.truncate_length - 1 if self.additional_target else self.truncate_length
        max_source_len = self.truncate_length - 1 if self.additional_source else self.truncate_length
        for l in open(data_fp).xreadlines():
            l = l.strip()
            src_tokens, tgt_tokens = map(lambda t: map(int, t.split(" ")), l.split(" ||| "))
            if (self.truncate_length and
                (len(src_tokens) > max_source_len or len(tgt_tokens) > max_target_len)):
                continue
            datas.append((src_tokens, tgt_tokens))
        if self.truncate_length:
            print "Truncated data length:", len(datas)
        if pick:
            rand = random.Random(3)
            rand.shuffle(datas)
            if self.avoid_valid_data:
                datas = datas[pick:]
            else:
                datas = datas[:pick]
        if self.sort_by_length:
            datas.sort(key=lambda d: len(d[0]))
        # Pack
        pack_size = self.batch_size * 100
        data_stream = open("/tmp/tmp_stream.pkl", "wb")
        for i in range(0, len(datas), pack_size):
            pack = datas[i:i + pack_size]
            StreamPickler.dump_one(pack, data_stream)
        data_stream.close()
        del datas
        # Process
        mp = Pool(cpu_count())
        for processed_batch_datas in mp.imap(self._process_batch, StreamPickler.load(open("/tmp/tmp_stream.pkl"))):
            sys.stdout.write(".")
            sys.stdout.flush()
            for batch_data in processed_batch_datas:
                StreamPickler.dump_one(batch_data, out)
        out.close()

    def _process_batch(self, sub_datas):
        processed_batches = []
        for i in range(0, len(sub_datas), self.batch_size):
            sub_data = sub_datas[i: i + self.batch_size]
            batch_input = []
            batch_target = []
            for src_tokens, tgt_tokens in sub_data:
                if self.additional_source:
                    src_tokens.insert(0, 0)
                if self.additional_target:
                    tgt_tokens.append(self.target_size - 1)
                if self.reverse_source:
                    src_tokens.reverse()
                batch_input.append(src_tokens)
                batch_target.append(tgt_tokens)
            batch_data = self._produce_batch(batch_input, batch_target)
            processed_batches.append(batch_data)
        return processed_batches


    def _produce_batch(self, batch_input, batch_target):
        self._pad_source(batch_input)
        target_mask = self._pad_target(batch_target)
        batch_input = np.array(batch_input)
        batch_target = np.array(batch_target)
        return batch_input, batch_target, target_mask

    def _pad_source(self, batch):
        max_len = max(map(len, batch))
        if self.fix_length:
            max_len = self.truncate_length
        for i in range(len(batch)):
            while len(batch[i]) < max_len:
                batch[i].insert(0, -1)

    def _pad_target(self, batch):
        mask = []
        max_len = max(map(len, batch))
        if self.fix_length:
            max_len = self.truncate_length
        for i in range(len(batch)):
            mask.append([1] * len(batch[i]) + [0] * (max_len - len(batch[i])))
            while len(batch[i]) < max_len:
                batch[i].append(0)
        return np.array(mask, dtype="float32")

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("output")
    ap.add_argument("--batch", default=128, type=int)
    ap.add_argument("--source_size", default=10000, type=int)
    ap.add_argument("--target_size", default=10000, type=int)
    ap.add_argument("--pick", default=0, type=int)
    ap.add_argument("--avoid_valid_data", action="store_true")
    ap.add_argument("--truncate_length", default=50, type=int)
    ap.add_argument("--fix_length", action="store_true")
    ap.add_argument("--reverse_source", action="store_true")
    args = ap.parse_args()


    packer = DataPacker(batch_size=args.batch, source_size=args.source_size,
                        target_size=args.target_size, avoid_valid_data=args.avoid_valid_data,
                        truncate_length=args.truncate_length, fix_length=args.fix_length,
                        reverse_source=args.reverse_source)
    packer.process(args.data, args.output, pick=args.pick)
