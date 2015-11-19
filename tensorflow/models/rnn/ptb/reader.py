# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile


def _read_words(filename):
  with gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: -x[1])[:9999]
  print(len(count_pairs))
  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))
  word_to_id['<UNK>'] = len(words)
  print(len(word_to_id))
  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  ids = []
  for word in data:
    if word in word_to_id:
      ids.append(word_to_id[word])
    else:
      ids.append(word_to_id['<UNK>'])
  return ids

def pos_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train_pos.txt")
  valid_path = os.path.join(data_path, "dev_pos.txt")
  test_path = os.path.join(data_path, "test_pos.txt")
  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)

  train_tag_path = os.path.join(data_path, "train_pos.tags")
  valid_tag_path = os.path.join(data_path, "dev_pos.tags")
  test_tag_path = os.path.join(data_path, "test_pos.tags")
  word_to_id_tags = _build_vocab(train_tag_path)
  train_data_tags = _file_to_word_ids(train_tag_path, word_to_id_tags)
  valid_data_tags = _file_to_word_ids(valid_tag_path, word_to_id_tags)
  test_data_tags = _file_to_word_ids(test_tag_path, word_to_id_tags)

  return (train_data, train_data_tags), (valid_data, valid_data_tags), (test_data,test_data_tags), vocabulary


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def length(data):
  return len(data[0])

def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data_txt = np.array(raw_data[0], dtype=np.int32)
  raw_data_tags = np.array(raw_data[1], dtype=np.int32)

  data_len = len(raw_data_txt)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  tags = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data_txt[batch_len * i:batch_len * (i + 1)]
    tags[i] = raw_data_tags[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = tags[:, i*num_steps:(i+1)*num_steps]
    yield (x, y)
