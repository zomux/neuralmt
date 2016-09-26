#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from itertools import izip, izip_longest
from vocab import NeuralVocab
from deepy import StreamPickler
import random

import logging
logging.basicConfig(level=logging.INFO)

class SequentialDataBuilder(object):


    def transform(self, vocab, data, additional_head=None, additional_tail=None, reverse=False):
        """
        Transform a list of data to indexes.
        :param vocab: `NeuralVocab` or the path of dump vocab file
        :type vocab: NeuralVocab or str
        :type data: list of list
        :param additional_head: append to head, e.g. `<s>`
        :param additional_tail: append to tail, e.g. `<s>`
        :param reverse
        """
        if type(vocab) == str:
            vocab = NeuralVocab(vocab)
        elif type(vocab) != NeuralVocab:
            raise SystemError(SequentialDataBuilder.__init__.__doc__)
        logging.info("vocab size: %d" % vocab.size())

        if additional_head and  not vocab.contains(additional_head):
            raise SystemError("%s is not in vocab" % additional_head)
        if additional_tail and not vocab.contains(additional_tail):
            raise SystemError("%s is not in vocab" % additional_tail)
        appending_head = vocab.encode_token(additional_head) if additional_head else None
        appending_tail = vocab.encode_token(additional_tail) if additional_tail else None
        # transform
        transformed_data = []
        for tokens in data:
            transformed_tokens = vocab.encode(tokens)
            if reverse:
                transformed_tokens.reverse()
            if additional_head:
                transformed_tokens.insert(0, appending_head)
            if additional_tail:
                transformed_tokens.append(appending_tail)
            transformed_data.append(transformed_tokens)
        return transformed_data
    def truncate(self, data_list, source_len=50, target_len=None):
        """
        Truncate a list of data.
        :param data_list:
        """
        if not target_len:
            target_len = source_len
        count_before_truncate = len(data_list[0])
        transformed_data_list = filter(lambda p: (not hasattr(p[0], "__len__") or len(p[0]) <= source_len) \
                                                 and (not hasattr(p[1], "__len__") or len(p[1]) <= target_len),
                                       zip(*data_list))
        logging.info("truncated data: %d -> %d" % (count_before_truncate, len(transformed_data_list)))
        return list(zip(*transformed_data_list))

    def make_batches(self, data, batch_size, output_mask=False, pad_value=0, fix_size=None, output_max_lens=False):
        """
        Make batch data.
        :type data: list of list
        :param batch_size
        :param output_mask: output the mask or not
        :param pad_value
        :param fix_size: fix batch size
        :param output_max_lens: output a list of max lengths
        """
        batches = []
        masks = []
        max_lens = []
        is_scalar = type(data[0]) in [int, float]
        for i in range(0, len(data), batch_size):
            sub_data = data[i: i + batch_size]
            if is_scalar:
                batches.append(sub_data)
            else:
                new_batch, new_mask = self._pad_batch(sub_data, pad_value, output_mask, fix_size=fix_size)
                batches.append(new_batch)
                if output_mask:
                    masks.append(new_mask)
                if output_max_lens:
                    max_lens.append(max(map(len, sub_data)))
        if not output_mask:
            masks = None

        if output_max_lens:
            return batches, masks, max_lens
        else:
            return batches, masks

    def _pad_batch(self, batch, pad_value, output_mask, fix_size=None):
        if fix_size:
            max_len = fix_size
        else:
            max_len = max(map(len, batch))
        mask = None
        if output_mask:
            mask = []
            for i in range(len(batch)):
                mask.append([1] * len(batch[i]) + [0] * (max_len - len(batch[i])))
            mask = np.array(mask, dtype="float32")
        new_batch = np.array(list(izip(*izip_longest(*batch, fillvalue=pad_value))))
        return new_batch, mask

    def dump(self, data_list, file_prefix, valid_batches=None, shuffle=True):
        """
        Dump data to pickle.
        :param data_list:
        :param file_prefix: prefix of output files
        :param valid_batches: size of valid data
        :param shuffle
        """
        if shuffle:
            for i in range(len(data_list)):
                random.Random(3).shuffle(data_list[i])
        if valid_batches:
            train_file = open("%s_train.pkl" % file_prefix, "wb")
            valid_file = open("%s_valid.pkl" % file_prefix, "wb")
        else:
            train_file = open("%s.pkl" % file_prefix, "wb")
            valid_file = None
        for i in range(len(data_list[0])):
            if valid_batches and i < valid_batches:
                StreamPickler.dump_one([d[i] for d in data_list], valid_file)
            else:
                StreamPickler.dump_one([d[i] for d in data_list], train_file)
        train_file.close()
        if valid_file:
            valid_file.close()



