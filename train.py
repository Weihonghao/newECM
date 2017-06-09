from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import datetime
import logging
import ECM_model

import numpy as np
import tensorflow as tf
from os.path import join as pjoin

import preprocess_data
from preprocess_data import EOS_ID
import utils
import seq2seq_model
from tensorflow.python.platform import gfile
import nltk
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

logging.basicConfig(level=logging.INFO)

#real_data_dir = "small_data"
real_data_dir = "data"

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99, "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 100000, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("testepochs", 10, "Number of epochs to test.")
tf.app.flags.DEFINE_float("keep_prob", 0.95, "Keep prob of output.")
tf.app.flags.DEFINE_integer("state_size", 256, "Size of encoder and decoder hidden layer.")
tf.app.flags.DEFINE_string("data_dir", real_data_dir + "/", "Data directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints/", "Checkpoint directory")
#tf.app.flags.DEFINE_string("checkpoint_dir", "checkpoints_backup/", "Checkpoint directory")
tf.app.flags.DEFINE_string("log_dir", "log/", "Tensorboard log directory")
tf.app.flags.DEFINE_string("vocab_path", real_data_dir + "/vocab.dat",
                           "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("embed_path", "",
                           "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_integer("max_train_data_size", 100,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_print", 1,
                            "How many training steps to print info.")
tf.app.flags.DEFINE_integer("steps_per_tensorboard", 200,
                            "How many training steps to write tensorboard.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("window_batch", 3, "window size / batch size")
tf.app.flags.DEFINE_string("retrain_embeddings", False, "Whether to retrain word embeddings")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def strip(x):
    return map(int, x.strip().split(" "))


'''def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]'''


class DataConfig(object):
    """docstring for DataDir"""

    def __init__(self, data_dir):
        self.train_from = pjoin(data_dir, 'train.ids.from')
        self.train_to = pjoin(data_dir, 'train.ids.to')
        self.train_tag = pjoin(data_dir, 'train.tag')

        self.val_from = pjoin(data_dir, 'val.ids.from')
        self.val_to = pjoin(data_dir, 'val.ids.to')
        self.val_tag = pjoin(data_dir, 'val.tag')
        self.test_from = pjoin(data_dir, 'test.ids.from')
        self.test_to = pjoin(data_dir, 'test.ids.to')
        self.test_tag = pjoin(data_dir, 'test.tag')


def read_data(data_config):
    train = []
    max_q_len = 0
    max_a_len = 0
    print("Loading training data from %s ..." % data_config.train_from)
    with gfile.GFile(data_config.train_from, mode="rb") as q_file, \
            gfile.GFile(data_config.train_to, mode="rb") as a_file, \
            gfile.GFile(data_config.train_tag, mode="rb") as t_file:
        for (q, a, t) in zip(q_file, a_file, t_file):
            question = strip(q)
            answer = strip(a)
            answer.append(EOS_ID)
            tag = strip(t)

            sample = [question, len(question), answer, len(answer), tag]
            train.append(sample)
            max_q_len = max(max_q_len, len(question))
            max_a_len = max(max_a_len, len(answer))

    print("Finish loading %d train data." % len(train))
    val = []
    print("Loading training data from %s ..." % data_config.val_from)
    with gfile.GFile(data_config.val_from, mode="rb") as q_file, \
            gfile.GFile(data_config.val_to, mode="rb") as a_file, \
            gfile.GFile(data_config.val_tag, mode="rb") as t_file:
        for (q, a, t) in zip(q_file, a_file, t_file):
            question = strip(q)
            answer = strip(a)
            answer.append(EOS_ID)
            tag = strip(t)

            sample = [question, len(question), answer, len(answer), tag]
            val.append(sample)
            max_q_len = max(max_q_len, len(question))
            max_a_len = max(max_a_len, len(answer))

    print("Finish loading %d val data." % len(val))
    print("Max question length %d " % max_q_len)
    print("Max answer length %d " % max_a_len)

    return {"training": train, "validation": val, "question_maxlen": max_q_len, "answer_maxlen": max_a_len}


def initialize_model(saver, session, model):
    """Create translation model and initialize or load parameters in session."""
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(FLAGS.checkpoint_dir, ckpt_name))
    else:
        print("Created model with fresh parameters.")
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


def save(saver, sess, step):
    model_name = "model"
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, model_name),
               global_step=step)


def train():
    """Train a en->fr translation model using WMT data."""
    start_time = time.time()
    data_config = DataConfig(FLAGS.data_dir)
    logFile = open('data/log.txt', 'w')
    embed_path = FLAGS.embed_path or pjoin(real_data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = utils.load_glove_embeddings(embed_path)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    non_emotion_size, vocab, rev_vocab = preprocess_data.initialize_vocabulary(vocab_path)
    FLAGS.vocab_size = len(vocab)
    FLAGS.non_emotion_size = non_emotion_size
    FLAGS.encoder_state_size = 128
    FLAGS.decoder_state_size = 2 * FLAGS.encoder_state_size
    print(embeddings.shape[0], len(vocab))
    assert embeddings.shape[0] == len(vocab)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:  # log_device_placement=True
        # Create model.
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        with tf.device('/gpu:1'):
            # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = ECM_model.ECMModel(embeddings, rev_vocab, FLAGS)
            saver = tf.train.Saver()
            initialize_model(saver, sess, model)

            '''tmpModel = initialize_model(sess, model)
            if tmpModel is not None:
                model = tmpModel'''
            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

            dataset = read_data(data_config)

            training_set = dataset['training']  # [question, len(question), answer, len(answer), tag]
            validation_set = dataset['validation']
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            for epoch in range(FLAGS.epochs):
                logging.info("=" * 10 + " Epoch %d out of %d " + "=" * 10, epoch + 1, FLAGS.epochs)
                batch_num = int(len(training_set) / FLAGS.batch_size)
                # prog = Progbar(target=batch_num)
                avg_loss = 0
                for i, batch in enumerate(
                        utils.minibatches(training_set, FLAGS.batch_size, window_batch=FLAGS.window_batch)):
                    global_batch_num = batch_num * epoch + i
                    time1 = time.time()
                    if global_batch_num % FLAGS.steps_per_tensorboard == FLAGS.steps_per_tensorboard - 1:
                        loss, merged = model.train(sess, batch, tensorboard=True)
                        summary_writer.add_summary(merged, global_batch_num)
                    else:
                        loss = model.train(sess, batch, tensorboard=False)
                    print('epoch %d [%d/%d], loss: %f' % (epoch, i, batch_num, loss))
                    # break
                    if global_batch_num % FLAGS.steps_per_checkpoint == FLAGS.steps_per_checkpoint - 1:
                        save(saver, sess, global_batch_num)
                    avg_loss += loss
                print('total time ' + str(datetime.timedelta(seconds=(time.time() - start_time))))
                print('average time ' + str(datetime.timedelta(seconds=((time.time() - start_time) / (epoch + 1)))))
                # break
                avg_loss /= batch_num
                logging.info("Average training loss: {}".format(avg_loss))

                co = 0
                logging.info("-- validation --")
                batch_num = len(validation_set) / FLAGS.batch_size
                avg_loss = 0
                for i, batch in enumerate(utils.minibatches(validation_set, FLAGS.batch_size,
                                                            window_batch=FLAGS.window_batch)):  # validation_set, FLAGS.batch_size, window_batch=FLAGS.window_batch)):
                    loss, ids = model.test(sess, batch)
                    print('loss: %f' % (loss))
                    print(batch[2].T)
                    print(ids)
                    if co < 10:
                        co += 1
                    else:
                        break

                        # avg_loss += loss
                        # avg_loss /= batch_num
                        # logging.info("Average validation loss: {}".format(avg_loss))


def test():
    start_time = time.time()
    data_config = DataConfig(FLAGS.data_dir)
    embed_path = FLAGS.embed_path or pjoin(real_data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = utils.load_glove_embeddings(embed_path)

    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    non_emotion_size, vocab, rev_vocab = preprocess_data.initialize_vocabulary(vocab_path)
    FLAGS.vocab_size = len(vocab)
    FLAGS.non_emotion_size = non_emotion_size
    FLAGS.encoder_state_size = 128
    FLAGS.decoder_state_size = 2 * FLAGS.encoder_state_size
    print(embeddings.shape[0], len(vocab))
    assert embeddings.shape[0] == len(vocab)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:  # log_device_placement=True
        # Create model.
        #sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        with tf.device('/gpu:1'):
            # print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
            model = ECM_model.ECMModel(embeddings, rev_vocab, FLAGS)
            saver = tf.train.Saver()
            initialize_model(saver, sess, model)

            tic = time.time()
            params = tf.trainable_variables()
            num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
            toc = time.time()
            logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

            dataset = read_data(data_config)

            training_set = dataset['training']  # [question, len(question), answer, len(answer), tag]
            validation_set = dataset['validation']
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
            avg_loss_list = []
            perplexity_list = []
            bleu_list = []
            for epoch in range(FLAGS.testepochs):
                logging.info("=" * 10 + " Epoch %d out of %d " + "=" * 10, epoch + 1, FLAGS.epochs)

                logging.info("-- validation --")
                batch_num = int(len(validation_set) / FLAGS.batch_size)
                avg_loss = 0
                perplexity = 0
                bleu = 0
                co = 0
                for i, batch in enumerate(utils.minibatches(validation_set, FLAGS.batch_size,
                                                            window_batch=FLAGS.window_batch)):  # validation_set, FLAGS.batch_size, window_batch=FLAGS.window_batch)):
                    loss, ids = model.test(sess, batch)
                    avg_loss += loss
                    perplexity += math.pow(2, loss)
                    print('loss: %f' % (loss))
                    #print(batch[2].T)
                    #print(ids)
                    for _ in range(len(ids)):
                        bleu += nltk.translate.bleu_score.sentence_bleu([batch[2].T[_]], ids[_])
                    co += len(ids)

                        # avg_loss += loss
                avg_loss = avg_loss * 1.0 / batch_num
                perplexity =  perplexity * 1.0 / co
                bleu = bleu * 1.0 / co
                avg_loss_list.append(avg_loss)
                perplexity_list.append(perplexity)
                bleu_list.append(bleu)

            logging.info("Average validation loss list: {}".format(avg_loss_list))
            logging.info("Average validation loss: {}".format(np.mean(np.array(avg_loss_list))))
            logging.info("Average validation perplexity list: {}".format(perplexity_list))
            logging.info("Average validation perplexity: {}".format(np.mean(np.array(perplexity_list))))
            logging.info("Average validation bleu list: {}".format(bleu_list))
            logging.info("Average validation bleu: {}".format(np.mean(np.array(bleu_list))))





def decode():
    embed_path = FLAGS.embed_path or pjoin(real_data_dir, "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    embeddings = utils.load_glove_embeddings(embed_path)
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    non_emotion_size, vocab, rev_vocab = preprocess_data.initialize_vocabulary(vocab_path)
    FLAGS.vocab_size = len(vocab)
    FLAGS.non_emotion_size = non_emotion_size
    FLAGS.encoder_state_size = 128
    FLAGS.decoder_state_size = 2 * FLAGS.encoder_state_size
    FLAGS.batch_size = 1
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model = ECM_model.ECMModel(embeddings, rev_vocab, FLAGS, forward_only=True)
        saver = tf.train.Saver()
        initialize_model(saver, sess, model)
        # Decode from standard input.
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            # Get token-ids for the input sentence.
            token_ids = preprocess_data.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab)
            print("> Choose Emotion [0-6]: ", end="")
            sys.stdout.flush()
            # Get emotion tag
            tag = sys.stdin.readline()
            tag = int(tf.compat.as_bytes(tag))
            data = [token_ids], [len(token_ids)], [tag]
            outputs = model.answer(sess, data)[0]
            # print("look here")
            # print(outputs)
            answer = ""
            for i in outputs:
                if i == preprocess_data.EOS_ID:
                    break
                else:
                    answer += rev_vocab[i] + " "
            print(answer)
            print("> ", end="")
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def main(_):
    if len(sys.argv) < 2:
        print("Usage: python train.py [train/test]")
    else:
        cmd = sys.argv[1]
        if cmd == "train":
            train()
        elif cmd == "test":
            test()
        elif cmd == "answer":
            decode()


if __name__ == "__main__":
    tf.app.run()
