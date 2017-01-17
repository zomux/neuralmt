#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download comtrans corpus with NLTK.
"""

import os
import logging

try:
    import nltk
except ImportError:
    raise Exception("You shall have nltk installed in order to download data.")

if "MT_ROOT" not in os.environ:
    raise Exception("Environment variable MT_ROOT is not found.")

if __name__ == '__main__':
    nltk.download("comtrans")
    text_root = "{}/text".format(os.environ["MT_ROOT"])
    if not os.path.exists(text_root):
        os.mkdir(text_root)
    print ("writing data ...")
    from nltk.corpus import comtrans
    fr_fp = open("{}/comtrans.de".format(text_root), "w")
    en_fp = open("{}/comtrans.en".format(text_root), "w")
    for pair in comtrans.aligned_sents():
        fr_fp.write(" ".join([w.encode("utf-8").lower() for w in pair.words]) + "\n")
        en_fp.write(" ".join([w.encode("utf-8").lower() for w in pair.mots]) + "\n")
    fr_fp.close()
    en_fp.close()
