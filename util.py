# Copyright 2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange
from flag import FLAGS
import random
from tensorflow.python.platform import gfile
import re

_PAD = b"<pad>"
_SOS = b"<sos>"
_EOS = b"<eos>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _SOS, _EOS, _UNK]

PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3

_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")


def tokenize(string):
    return [int(s) for s in string.split()]


def tokenize_x(string):
    return [[int(ele) for ele in sen.split()] for sen in string.split('\t')]


def pair_iter(fnamex, fnamey, batch_size, max_seq_length, max_num_wit, sort_and_shuffle=True):
    fdx, fdy = open(fnamex), open(fnamey)
    batches = []

    while True:
        if len(batches) == 0:
            refill(batches, fdx, fdy, batch_size, max_seq_length, max_num_wit, sort_and_shuffle=sort_and_shuffle)
        if len(batches) == 0:
            break

        x_tokens, y_tokens = batches.pop(0)
        y_tokens = add_sos_eos(y_tokens)
        x_padded, y_padded = padded_x(x_tokens), padded(y_tokens)

        source_tokens = np.array(x_padded)
        source_tokens = np.transpose(source_tokens, [2, 1, 0])
        source_mask= (source_tokens != PAD_ID).astype(np.int32)
        target_tokens = np.array(y_padded).T
        target_mask = (target_tokens != PAD_ID).astype(np.int32)
        yield (source_tokens, source_mask, target_tokens, target_mask)

    return


def refill(batches, fdx, fdy, batch_size, max_seq_len, max_num_wit,
           sort_and_shuffle=True):
    line_pairs = []
    linex, liney = fdx.readline(), fdy.readline()

    while linex and liney:
        x_tokens, y_tokens = tokenize_x(linex.strip()), tokenize(liney.strip())
        x_tokens = [ele[:max_seq_len] for ele in x_tokens][:max_num_wit]
        if len(x_tokens) > 0 and len(y_tokens) < FLAGS.max_seq_len:
            line_pairs.append((x_tokens, y_tokens))
        linex, liney = fdx.readline(), fdy.readline()

    if sort_and_shuffle:
        line_pairs = sorted(line_pairs, key=lambda e: len(e[0]))
        num_wit = [len(ele) for ele in line_pairs]
        list_num_wit = [[] for _ in range(1, max(num_wit))]
        num_line = len(num_wit)
        for i in range(num_line):
            list_num_wit[num_wit[i]].append(line_pairs[i])
        list_num_wit = [ele for ele in list_num_wit if len(ele) > 0]
        list_num_wit = map(lambda tokenlist:
                           sorted(tokenlist,
                                  key=lambda e: max([len(ele) for ele in e])),
                           list_num_wit)
        line_pairs = [item for l in list_num_wit for item in l]

    for batch_start in xrange(0, len(line_pairs), batch_size):
        x_batch, y_batch = zip(*line_pairs[batch_start:batch_start+batch_size])
        batches.append((x_batch, y_batch))

    if sort_and_shuffle:
        random.shuffle(batches)
    return


def add_sos_eos(tokens):
    return map(lambda token_list: [SOS_ID] + token_list + [EOS_ID], tokens)


def padded(tokens):
    maxlen = max(map(lambda x: len(x), tokens))
    return map(lambda token_list: token_list + [PAD_ID] * (maxlen - len(token_list)), tokens)


def padded_x(tokens):
    max_num_wit = max(map(lambda x: len(x), tokens))
    max_seq_len = max(map(lambda x: max([len(ele) for ele in x]), tokens))
    tokens = [map(lambda token_list: token_list +
                                     [PAD_ID] * (max_seq_len
                                                 - len(token_list)),
                  ele) for ele in tokens]
    return map(lambda token_list: token_list + [PAD_ID] * (max_num_wit - len(token_list)), tokens)


def initialize_vocabulary(vocabulary_path, bpe=False):
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        # Call ''.join below since BPE outputs split pairs with spaces
        if bpe:
            vocab = dict([(''.join(x.split(' ')), y) for (y, x) in enumerate(rev_vocab)])
        else:
            vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)

