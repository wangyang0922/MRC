# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import jieba
import pandas as pd
import nltk
import numpy as np
from collections import defaultdict
from utils.file_utils import save_dict
from utils.multi_proc_utils import parallelize, cores
from utils.config import search_dev_data_path, zhidao_dev_data_path, merger_dev_seg_path,stop_word_path,save_wv_model_path,vocab_path,embedding_matrix_path,search_train_data_path, merger_seg_path,zhidao_train_data_path, search_test_data_path, zhidao_test_data_path,embedding_dim,wv_train_epochs
from gensim.models.word2vec import LineSentence, Word2Vec
# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def build_dataset(search_dev_data_path, zhidao_dev_data_path):
    '''
    数据加载+预处理
    :param search_dev_data_path:search集路径
    :param zhidao_dev_data_path: zhidao集路径
    :return: 合并后的数据
    '''
    # 1.加载数据
    search_dev_df = pd.read_json(search_dev_data_path, lines=True)
    zhidao_dev_df = pd.read_json(zhidao_dev_data_path,encoding='utf-8', lines=True)
    print('search dev data size {},zhidao dev data size {}'.format(len(search_dev_df), len(zhidao_dev_df)))
    # print(search_dev_data.columns)

    search_dev_df['answers'] = search_dev_df[['answers']].apply(sentence_proc, axis=1)
    search_dev_df['entity_answers'] = search_dev_df[['entity_answers']].apply(sentences_proc, axis=1)
    search_dev_df['documents'] = search_dev_df[['documents']].apply(documents_proc, axis=1)
    zhidao_dev_df['answers'] = zhidao_dev_df[['answers']].apply(sentence_proc, axis=1)
    zhidao_dev_df['entity_answers'] = zhidao_dev_df[['entity_answers']].apply(sentences_proc, axis=1)
    zhidao_dev_df['documents'] = zhidao_dev_df[['documents']].apply(documents_proc, axis=1)

    # print(search_dev_df["documents"])
    # print(search_dev_df['entity_answers'])
    # print(search_dev_df['question'])
    # print(search_dev_df['answers'])
    # print(zhidao_dev_df["documents"])
    # print(zhidao_dev_df['entity_answers'])
    # print(zhidao_dev_df['question'])
    # print(zhidao_dev_df['answers'])


    # 3.多线程, 批量数据处理
    search_dev_df = parallelize(search_dev_df, split_sentences_proc)
    zhidao_dev_df = parallelize(zhidao_dev_df, split_sentences_proc)

    # 4. 合并训练测试集合
    search_dev_df['merged'] = search_dev_df[['documents', 'entity_answers', 'question', 'answers']].apply(lambda x: ' '.join(x), axis=1)
    zhidao_dev_df['merged'] = search_dev_df[['documents', 'entity_answers', 'question', 'answers']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([search_dev_df[['merged']], zhidao_dev_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(search_dev_df),
                                                                               len(zhidao_dev_df),
                                                                               len(merged_df)))
    # 6. 保存合并数据
    merged_df.to_csv(merger_dev_seg_path, index=None, header=False)

    # 7. 训练词向量
    print('start build w2v model')
    wv_model = Word2Vec(LineSentence(merger_dev_seg_path),
                        size=embedding_dim,
                        sg=1,
                        workers=cores,
                        iter=wv_train_epochs,
                        window=5,
                        min_count=5)
    # 8. 填充开始结束符号,未知词填充 oov, 长度填充
    # 使用GenSim训练得出的vocab
    vocab = wv_model.wv.vocab

    # 9、保存字典
    save_dict(vocab_path, vocab)

    # 10、保存词向量模型
    wv_model.save(save_wv_model_path)




    return
def split_sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 用词清除无
    # sentence = clean_sentence(sentence)
    # 分段切词
    sentence = seg_proc(sentence)
    # 过滤停用词
    words = filter_words(sentence)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)

def en_split_sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 用词清除无
    # sentence = clean_sentence(sentence)
    # 分段切词
    words = cut_sentence(sentence)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)

def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)

remove_words = ['|', '[', ']', '语音', '图片']

def filter_words(sentence):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    words = sentence.split(' ')
    # 去掉多余空字符
    words = [word for word in words if word and word not in remove_words]
    # 去掉停用词 包括一下标点符号也会去掉
    words = [word for word in words if word not in stop_words]
    return words


def seg_proc(sentence):
    tokens = sentence.split('|')
    result = []
    for t in tokens:
        result.append(cut_sentence(t))
    return ' | '.join(result)

def cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = jieba.cut(line)
    return ' '.join(tokens)

def en_cut_sentence(line):
    # 切词，默认精确模式，全模式cut参数cut_all=True
    tokens = nltk.word_tokenize(line)
    return ' '.join(tokens)

def split_sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['question', 'answers', 'entity_answers', 'documents']:
        df[col_name] = df[col_name].apply(split_sentence_proc)
    return df

def en_split_sentences_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['merge']:
        df[col_name] = df[col_name].apply(en_split_sentence_proc)
    return df
def sentences_proc(df):
    df.fillna('', inplace=True)
    res = df.apply(lambda x: [' | '.join(data) for data in x])
    return sentence_proc(res)


def documents_proc(df):

    return df.apply(lambda x: document_proc(x))

def document_proc(datas):
    res = ''
    for data in datas:
        res += data["title"] + ' | '.join(data["paragraphs"])
    return res

def sentence_proc(df):
    return df.apply(lambda x: ' | '.join(x))

def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            w, i = line
            f.write("%s\t%d\n" % (w, i))

def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)

        for i in items:
            print(i)
            # for i in item.split(" "):
            #     i = i.strip()
            #     if not i: continue
            #     i = i if not lower else item.lower()
            #     dic[i] += 1
        # sort
        """
        按照字典里的词频进行排序，出现次数多的排在前面
        your code(one line)
        """
        dic = sorted(dic.items(), key=lambda item: item[1], reverse=True)

        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)

    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    your code
    vocab = (one line)
    reverse_vocab = (one line)
    """


    vocab = []
    reverse_vocab = []
    for i, item in enumerate(result):
        vocab.append((item, i))
        reverse_vocab.append((i, item))

    vocab = [(item,i) for i, item in enumerate(result)]
    reverse_vocab = [(i,item) for i, item in enumerate(result)]

    return vocab, reverse_vocab

if __name__ == '__main__':
    # 数据集批量处理
    en_cut_sentence("i am a boy")
    # build_dataset(search_dev_data_path, zhidao_dev_data_path)
