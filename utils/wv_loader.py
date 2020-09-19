# -*- coding:utf-8 -*-
# Created by LuoJie at 11/22/19
from gensim.models.word2vec import Word2Vec
import numpy as np
# 引入日志配置

from utils.config import embedding_matrix_path, vocab_path, save_wv_model_path


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'
    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]
    MASK_COUNT = len(MASKS)

    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)
    START_DECODING_INDEX = MASKS.index(START_DECODING)
    STOP_DECODING_INDEX = MASKS.index(STOP_DECODING)

    def __init__(self, vocab_file=vocab_path, vocab_max_size=None):
        """
        Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param vocab_max_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2id)

    def load_vocab(self, file_path, vocab_max_size=None):
        """
        读取字典
        :param file_path: 文件路径
        :return: 返回读取后的字典
        """
        vocab = {}

        reverse_vocab = {}

        for line in open(file_path, "r", encoding='utf-8').readlines():
            word, index = line.strip().split("\t")
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index >= vocab_max_size:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index
            reverse_vocab[index] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.UNKNOWN_TOKEN_INDEX
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            return self.UNKNOWN_TOKEN
        return self.id2word[word_id]

    def size(self):
        return self.count


def load_embedding_matrix(filepath=embedding_matrix_path, max_vocab_size=50000):
    """
    加载 embedding_matrix_path
    """
    embedding_matrix = np.load(filepath + '.npy')
    return embedding_matrix[:max_vocab_size]


def load_word2vec_file():
    # 保存词向量模型
    return Word2Vec.load(save_wv_model_path)


if __name__ == '__main__':
    # vocab 对象
    # vocab = Vocab(vocab_path)
    # print(vocab.count)
    print(load_embedding_matrix(max_vocab_size=300))
