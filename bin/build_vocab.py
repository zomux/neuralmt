#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import Counter
import cPickle as pickle

RESERVED_VOCAB = ["<s>", "UNK"]
UNKPOS_VOCAB = ["UNKPOS%d" % i for i in range(-7, 8)]

BILINGUAL_DELIMITER = " ||| "

class VocabBuilder(object):

    def __init__(self, unkpos=False, char_based=False):
        self.vocab = None
        self.unkpos = unkpos
        self.char_based = char_based

    def build(self, fp, limit=10000, side=None):
        side = side if side in ["source", "target"] else None
        all_tokens = []
        for l in open(fp).xreadlines():
            l = l.strip()
            if not l: continue
            if side and BILINGUAL_DELIMITER in l:
                pair = l.split(BILINGUAL_DELIMITER)
                line = pair[0] if side == "source" else pair[1]
            else:
                line = l
            if self.char_based:
                tokens = line
            else:
                tokens = line.split(" ")
            all_tokens.extend(tokens)
        token_counter = Counter(all_tokens)
        common_tokens = [t[0] for t in token_counter.most_common()]
        if self.unkpos:
            common_tokens = RESERVED_VOCAB + UNKPOS_VOCAB + common_tokens
        else:
            common_tokens = RESERVED_VOCAB + common_tokens
        self.vocab = common_tokens[:limit]


    def save_pickle(self, path):
        if not self.vocab:
            raise Exception("No vocab")
        print "Dump vocab, length = %d" % len(self.vocab)
        pickle.dump(self.vocab, open(path, "wb"))

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("save")
    ap.add_argument("--limit", default=10000, type=int)
    ap.add_argument("--unkpos", action="store_true")
    ap.add_argument("--side", help="[source|target] if the data is a parallel corpus separated by ' ||| '")
    ap.add_argument("--character", action="store_true", help="turn on this flag for a character-based model")
    args = ap.parse_args()

    vocab = VocabBuilder(unkpos=args.unkpos, char_based=args.character)
    vocab.build(args.data, args.limit, args.side)
    vocab.save_pickle(args.save)
