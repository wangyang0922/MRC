import json
import os
import pickle
from collections import deque

import numpy as np
from tqdm import tqdm


def _load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, embedding_matrix_file):
    if os.path.exists(embedding_matrix_file):
        print('loading embedding_matrix:', embedding_matrix_file)
        embedding_matrix = pickle.load(open(embedding_matrix_file, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        # fname = './data、glove/glove.42B.300d.txt'
        fname = './data/glove/glove.6B.50d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file, 'wb'))
    return embedding_matrix


def mytqdm(list_, desc="", show=True):
    if show:
        pbar = tqdm(list_)
        pbar.set_description(desc)
        return pbar
    return list_


def json_pretty_dump(obj, fh):
    return json.dump(obj, fh, sort_keys=True, indent=2, separators=(',', ': '))


def index(l, i):
    return index(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]


def fill(l, shape, dtype=None):
    out = np.zeros(shape, dtype=dtype)
    stack = deque()
    stack.appendleft(((), l))
    while len(stack) > 0:
        indices, cur = stack.pop()
        if len(indices) < shape:
            for i, sub in enumerate(cur):
                stack.appendleft([indices + (i,), sub])
        else:
            out[indices] = cur
    return out


def short_floats(o, precision):
    class ShortFloat(float):
        def __repr__(self):
            return '%.{}g'.format(precision) % self

    def _short_floats(obj):
        if isinstance(obj, float):
            return ShortFloat(obj)
        elif isinstance(obj, dict):
            return dict((k, _short_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(map(_short_floats, obj))
        return obj

    return _short_floats(o)


def argmax(x):
    return np.unravel_index(x.argmax(), x.shape)
