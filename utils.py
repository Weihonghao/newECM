import logging
import numpy as np
from os.path import join as pjoin
from tensorflow.python.platform import gfile

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]



def load_glove_embeddings(embed_path):
    logger.info("Loading glove embedding...")
    glove = np.load(embed_path)['glove']
    logger.info("Dimension: {}".format(glove.shape[1]))
    logger.info("Vocabulary: {}" .format(glove.shape[0]))
    return glove

def add_paddings(sentence, max_length, n_features=1):
    mask = [True] * len(sentence)
    pad_len = max_length - len(sentence)
    if pad_len > 0:
        padded_sentence = sentence + [0] * pad_len
        mask += [False] * pad_len
    else:
        padded_sentence = sentence[:max_length]
        mask = mask[:max_length]
    return padded_sentence, mask

def preprocess_dataset(dataset, question_maxlen, context_maxlen):
    processed = []
    for q, q_len, c, c_len, ans in dataset:
        # add padding:
        q_padded, q_mask = add_paddings(q, question_maxlen)
        c_padded, c_mask = add_paddings(c, context_maxlen)
        processed.append([q_padded, q_mask, c_padded, c_mask, ans])
    return processed

def strip(x):
    return map(int, x.strip().split(" "))

def get_minibatches(data, minibatch_size, shuffle=True):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatch_start in np.arange(0, data_size, minibatch_size):
        minibatch_indices = indices[minibatch_start:minibatch_start + minibatch_size]
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)


def get_minibatches_with_window(data, batch_size, window_batch):
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    batch_num = int(np.ceil(data_size * 1.0 / batch_size))
    window_size = min([batch_size*window_batch, data_size])
    window_start = np.random.randint(data_size-window_size+1, size=(batch_num,))
    # print(window_start)
    for i in range(batch_num):
        window_index = np.arange(window_start[i], window_start[i]+window_size)
        # print(window_index)
        minibatch_indices = np.random.choice(window_index,size = (batch_size,),replace=False)
        # print(minibatch_indices)
        yield [minibatch(d, minibatch_indices) for d in data] if list_data \
            else minibatch(data, minibatch_indices)

def minibatches(data, batch_size, shuffle=True, window_batch=None):
    batches = [np.array(col) for col in zip(*data)]
    if window_batch is None:
        return get_minibatches(batches, batch_size, shuffle)
    else:
        return get_minibatches_with_window(batches, batch_size, window_batch)
