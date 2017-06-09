
"""Utilities for downloading data from WMT, tokenizing, vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf
from tqdm import *
import numpy as np
from os.path import join as pjoin

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
_DIGIT_RE = re.compile(br"\d")

def setup_args():
    parser = argparse.ArgumentParser()
    home = os.getcwd()
    #vocab_dir = os.path.join(home, "data")
    #glove_dir = os.path.join(home, "data", "glove")
    #source_dir = os.path.join(home, "data")
    vocab_dir = os.path.join(home, "medium_data")
    glove_dir = os.path.join(home, "medium_data", "glove")
    source_dir = os.path.join(home, "medium_data")
    #vocab_dir = os.path.join(home, "small_data")
    #glove_dir = os.path.join(home, "small_data", "glove")
    #source_dir = os.path.join(home, "small_data")
    parser.add_argument("--source_dir", default=source_dir)
    parser.add_argument("--glove_dir", default=glove_dir)
    parser.add_argument("--vocab_dir", default=vocab_dir)
    parser.add_argument("--glove_dim", default=300, type=int)
    parser.add_argument("--random_init", default=True, type=bool)
    parser.add_argument("--word_cnt_threshold", default=100, type=int)
    return parser.parse_args()

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, emotion_vocab_path, data_paths, threshold=0, tokenizer=None):
    # if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s (emotion vocab %s, th = %d) from data %s" \
         % (vocabulary_path, emotion_vocab_path, threshold, str(data_paths)))
    emotion_vocab = {}
    if gfile.Exists(emotion_vocab_path):
      with gfile.GFile(emotion_vocab_path, mode="rb") as ef:
        for line in ef.readlines():
          w = line.strip()
          emotion_vocab[w] = 0
    else:
      raise ValueError("Emotion vocabulary file %s not found.", emotion_vocab_path)

    non_emotion_vocab = {}
    for path in data_paths:
        with open(path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    if w in emotion_vocab:
                        emotion_vocab[w] += 1
                    else:
                        if w in non_emotion_vocab:
                            non_emotion_vocab[w] += 1
                        else:
                            non_emotion_vocab[w] = 1
    if threshold > 0:
      non_emotion_vocab = {k: v for k, v in non_emotion_vocab.iteritems() if v > threshold}
    vocab_list = _START_VOCAB + sorted(non_emotion_vocab, key=non_emotion_vocab.get, reverse=True) \
                 + sorted(emotion_vocab, key=emotion_vocab.get, reverse=True)
    non_emotion_size = len(_START_VOCAB) + len(non_emotion_vocab)
    print("Non emotion vocabulary size: %d" % non_emotion_size)
    print("Emotion vocabulary size: %d" % len(emotion_vocab))
    print("Total vocabulary size: %d" % len(vocab_list))
    with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        vocab_file.write(str(non_emotion_size) + b"\n")
        for w in vocab_list:
            vocab_file.write(w + b"\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      non_emotion_size = int(f.readline().strip())
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return non_emotion_size, vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  # if not gfile.Exists(target_path):
  print("Tokenizing data in %s" % data_path)
  _, vocab, _ = initialize_vocabulary(vocabulary_path)
  with gfile.GFile(data_path, mode="rb") as data_file:
    with gfile.GFile(target_path, mode="w") as tokens_file:
      counter = 0
      for line in data_file:
        counter += 1
        if counter % 100000 == 0:
          print("  tokenizing line %d" % counter)
        token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab,
                                          tokenizer, normalize_digits)
        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def process_glove(args, vocab_list, save_path, size=4e5, random_init=True):
    """
    :param vocab_list: [vocab]
    :return:
    """

    if not gfile.Exists(save_path + ".npz"):
        glove_path = os.path.join(args.glove_dir, "glove.6B.{}d.txt".format(args.glove_dim))
        if random_init:
            glove = np.random.randn(len(vocab_list), args.glove_dim)
        else:
            glove = np.zeros((len(vocab_list), args.glove_dim))
        found = 0

        vocab_dict = dict(zip(vocab_list, range(len(vocab_list))))


        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_dict:
                    idx = vocab_dict[word]
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_dict:
                    idx = vocab_dict[word.capitalize()]
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_dict:
                    idx = vocab_dict[word.upper()]
                    glove[idx, :] = vector
                    found += 1

        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def prepare_data():
    args = setup_args()
    emotion_vocab_path = pjoin(args.vocab_dir, "emotion_vocab.txt")
    vocab_path = pjoin(args.vocab_dir, "vocab.dat")
    train_path = pjoin(args.source_dir, "train")
    valid_path = pjoin(args.source_dir, "val")
    test_path = pjoin(args.source_dir, "test")

    create_vocabulary(vocab_path, emotion_vocab_path,
                      [pjoin(args.source_dir, "train.from"),
                       pjoin(args.source_dir, "train.to"),
                       pjoin(args.source_dir, "val.to"),
                       pjoin(args.source_dir, "val.from"),
                       ], threshold=args.word_cnt_threshold)
    non_emotion_size, vocab, rev_vocab = initialize_vocabulary(pjoin(args.vocab_dir, "vocab.dat"))

    process_glove(args, rev_vocab, args.source_dir + "/glove.trimmed.{}".format(args.glove_dim),
                    random_init=args.random_init)

    # ======== Creating Dataset =========
    x_train_ids_path = train_path + ".ids.from"
    y_train_ids_path = train_path + ".ids.to"
    data_to_token_ids(train_path + ".from", x_train_ids_path, vocab_path)
    data_to_token_ids(train_path + ".to", y_train_ids_path, vocab_path)

    x_ids_path = valid_path + ".ids.from"
    y_ids_path = valid_path + ".ids.to"
    data_to_token_ids(valid_path + ".from", x_ids_path, vocab_path)
    data_to_token_ids(valid_path + ".to", y_ids_path, vocab_path)



if __name__ == '__main__':
  prepare_data()
