# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib

# 预处理数据 构建数据集
is_build_dataset = True

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent
# 全部数据路径
search_dev_data_path = os.path.join(root, 'data', 'search.dev.json')
zhidao_dev_data_path = os.path.join(root, 'data', 'zhidao.dev.json')

# 训练数据路径
search_train_data_path = os.path.join(root, 'data', 'search.train.json')
zhidao_train_data_path = os.path.join(root, 'data', 'zhidao.train.json')

# 测试数据路径
search_test_data_path = os.path.join(root, 'data', 'search.test.json')
zhidao_test_data_path = os.path.join(root, 'data', 'zhidao.test.json')

# 合并分词
merger_seg_path = os.path.join(root, 'data', 'merged_seg_data.csv')
merger_dev_seg_path = os.path.join(root, 'data', 'merged_dev_seg_data.csv')

# 停用词路径
# stop_word_path = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')
stop_word_path = os.path.join(root, 'data', 'stopwords/stopwords.txt')

# 自定义切词表
user_dict = os.path.join(root, 'data', 'user_dict.txt')

# 词向量路径
save_wv_model_path = os.path.join(root, 'data', 'wv', 'word2vec.model')
# 词向量矩阵保存路径
embedding_matrix_path = os.path.join(root, 'data', 'wv', 'embedding_matrix')
# 字典路径
vocab_path = os.path.join(root, 'data', 'wv', 'vocab.txt')
reverse_vocab_path = os.path.join(root, 'data', 'wv', 'reverstest_save_dire_vocab.txt')

# 词向量训练轮数
wv_train_epochs = 1

# 模型保存文件夹
# checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_pgn_cov_not_clean')

checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'pgn')

# checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')
seq2seq_checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'training_checkpoints_seq2seq')
transformer_checkpoint_dir = os.path.join(root, 'data', 'checkpoints', 'transformer')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')

# 结果保存文件夹
save_result_dir = os.path.join(root, 'result')

# 词向量维度
embedding_dim = 300

sample_total = 82871

batch_size = 64

epochs = 2

vocab_size = 50000
