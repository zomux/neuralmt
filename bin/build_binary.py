#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from argparse import ArgumentParser
import cPickle as pickle

UNKPOS_LIMIT = 7

def null_generator():
    while True:
        yield "0-0"

class UNKBuilder(object):

    def __init__(self, src_vocab_pkl, tgt_vocab_pkl, unkpos=True, char_based=False):
        self.source_vocab = pickle.load(open(src_vocab_pkl))
        self.target_vocab = pickle.load(open(tgt_vocab_pkl))
        self.source_vocab_set = set(self.source_vocab)
        self.target_vocab_set = set(self.target_vocab)
        self.unkpos = unkpos
        self.char_based = char_based

    def process(self, data, align_fp, output_binary=False):
        progress = 0
        lines = open(data).xreadlines()
        align_lines = open(align_fp).xreadlines() if align_fp else null_generator()
        for l, align_l in zip(lines, align_lines):
            progress += 1
            src_l, tgt_l = l.strip().split(" ||| ")
            src_l, tgt_l, align_l = map(str.strip, [src_l, tgt_l, align_l])
            if not src_l or not tgt_l:
                continue
            if self.char_based:
                src_tokens = src_l
                tgt_tokens = tgt_l
            else:
                src_tokens = src_l.split(" ")
                tgt_tokens = tgt_l.split(" ")
            src_tokens = [t if t in self.source_vocab_set else "UNK" for t in src_tokens]
            tgt_tokens = [t if t in self.target_vocab_set else "UNK" for t in tgt_tokens]
            align_map = self.build_align_map(align_l)
            # Replace for unkpos
            if self.unkpos and "UNK" in tgt_tokens:
                new_tgt_tokens = []
                for tgt_i in range(len(tgt_tokens)):
                    if tgt_tokens[tgt_i] == "UNK":
                        unkpos_failed = True
                        if tgt_i in align_map:
                            src_idx_list = align_map[tgt_i]
                            for src_idx in src_idx_list:
                                offset = src_idx - tgt_i
                                if abs(offset) <= UNKPOS_LIMIT:
                                    new_tgt_tokens.append("UNKPOS%d" % offset)
                                    unkpos_failed = False
                        if unkpos_failed:
                            new_tgt_tokens.append("UNK")

                    else:
                        new_tgt_tokens.append(tgt_tokens[tgt_i])
                tgt_tokens = new_tgt_tokens
            ###
            if not output_binary:
                print "%s ||| %s" % (" ".join(src_tokens), " ".join(tgt_tokens))
            else:
                src_bins = [str(self.source_vocab.index(t)) for t in src_tokens]
                tgt_bins = [str(self.target_vocab.index(t)) for t in tgt_tokens]
                print "%s ||| %s" % (" ".join(src_bins), " ".join(tgt_bins))
            if progress % 10000 == 0:
                sys.stderr.write(".")
                sys.stderr.flush()

    def build_align_map(self, align_text):
        align_map = {}
        for align_pair in align_text.split(" "):
            src_pos, tgt_pos = map(int, align_pair.split("-"))
            if tgt_pos not in align_map:
                align_map[tgt_pos] = []
            align_map[tgt_pos].append(src_pos)
        return align_map


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("source_vocab")
    ap.add_argument("target_vocab")
    ap.add_argument("--align")
    ap.add_argument("--unkpos", action="store_true")
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--character", action="store_true")
    args = ap.parse_args()

    builder = UNKBuilder(args.source_vocab, args.target_vocab, char_based=args.character)
    if args.unkpos:
        align_path = args.align
    else:
        align_path = ""
    builder.process(args.data, align_path, output_binary=args.binary)
