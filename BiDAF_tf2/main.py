import os

from config import test_data_pickle_data_path, train_data_pickle_data_path, max_char_len
from utils import build_embedding_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # cpu

import logging
import warnings

from tensorflow.keras.layers import concatenate

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import pickle

print("tf.__version__:", tf.__version__)


def dump_pkl(vocab, pkl_path, overwrite=True):
    """
    存储文件
    :param pkl_path:
    :param overwrite:
    :return:
    """
    if pkl_path and os.path.exists(pkl_path) and not overwrite:
        return
    if pkl_path:
        with open(pkl_path, 'wb') as f:
            pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
            # pickle.dump(vocab, f, protocol=0)
        print("save %s ok." % pkl_path)


def load_pkl(pkl_path):
    """
    加载词典文件
    :param pkl_path:
    :return:
    """
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    return result


class BiDAF:

    def __init__(
            self, clen, qlen,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
            glove_weight=None,
            char_vocab_size=0
    ):
        """
        双向注意流模型
        :param clen:context 长度
        :param qlen: question 长度
        :param emb_size: 词向量维度
        :param max_features: 词汇表最大数量
        :param num_highway_layers: 高速神经网络的个数 2
        :param encoder_dropout: encoder dropout 概率大小
        :param num_decoders:解码器个数
        :param decoder_dropout: decoder dropout 概率大
        """
        self.clen = clen
        self.qlen = qlen

        self.word_vocab_size = glove_weight.shape[0]
        self.emb_size = glove_weight.shape[1]
        self.char_vocab_size = char_vocab_size
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout
        self.kernel_size = [2, 3, 4]
        # ADD
        self.glove_weight = glove_weight

        self.conv1d_mapper = {
            size: tf.keras.layers.Conv1D(filters=2, kernel_size=size, strides=1)
            for size in self.kernel_size
        }
        self.pool_mapper = {
            size: tf.keras.layers.MaxPool1D(pool_size=max_char_len - size + 1)
            for size in self.kernel_size
        }

        # self.conv1d_layer = tf.keras.layers.Conv1D(filters=2, kernel_size=self.kernel_size, strides=1)
        # self.maxpool1d_layer = tf.keras.layers.MaxPool1D(pool_size=max_char_len - self.kernel_size + 1)

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # TODO：homework：使用glove word embedding（或自己训练的w2v） 和 CNN char embedding

        word_embedding_layer = tf.keras.layers.Embedding(self.word_vocab_size, self.emb_size,
                                                         weights=[self.glove_weight],
                                                         trainable=False)

        char_embedding_layer = tf.keras.layers.Embedding(self.char_vocab_size, self.emb_size,
                                                         embeddings_initializer='uniform')
        # char
        # (None, 30, 10)
        cinn_char = tf.keras.layers.Input(shape=(self.clen, max_char_len,), name='context_input_char')
        qinn_char = tf.keras.layers.Input(shape=(self.qlen, max_char_len,), name='question_input_char')

        # word
        # (None, 30)
        cinn_word = tf.keras.layers.Input(shape=(self.clen,), name='context_input_word')
        qinn_word = tf.keras.layers.Input(shape=(self.qlen,), name='question_input_word')

        # word
        # (None, 30, 50)
        cemb = word_embedding_layer(cinn_word)
        # (None, 30, 50)
        qemb = word_embedding_layer(qinn_word)

        # char feature
        # (None, 30, 10, 50)
        c_char_emb = char_embedding_layer(cinn_char)
        # (None, 30, 10, 50)
        q_char_emb = char_embedding_layer(qinn_char)

        # (None, 30, 6)
        cemb_c = self.multi_conv1d(c_char_emb)
        qemb_q = self.multi_conv1d(q_char_emb)

        # (None, 30, 56)
        cemb = tf.concat([cemb, cemb_c], axis=2)
        qemb = tf.concat([qemb, qemb_q], axis=2)

        for i in range(self.num_highway_layers):
            """
            使用两层高速神经网络
            """
            highway_layer = layers.Highway(name=f'Highway{i}')
            chighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'CHighway{i}')
            qhighway = tf.keras.layers.TimeDistributed(highway_layer, name=f'QHighway{i}')
            cemb = chighway(cemb)
            qemb = qhighway(qemb)

        # 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(cemb)  # 编码context
        qencode = encoder_layer(qemb)  # 编码question

        # 3.注意流层
        similarity_layer = layers.Similarity(name='SimilarityLayer')
        similarity_matrix = similarity_layer([cencode, qencode])

        c2q_att_layer = layers.C2QAttention(name='C2QAttention')
        q2c_att_layer = layers.Q2CAttention(name='Q2CAttention')

        c2q_att = c2q_att_layer(similarity_matrix, qencode)
        q2c_att = q2c_att_layer(similarity_matrix, cencode)

        # 上下文嵌入向量的生成
        merged_ctx_layer = layers.MergedContext(name='MergedContext')
        merged_ctx = merged_ctx_layer(cencode, c2q_att, q2c_att)

        # 4.模型层
        modeled_ctx = merged_ctx
        for i in range(self.num_decoders):
            decoder_layer = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    self.emb_size,
                    recurrent_dropout=self.decoder_dropout,
                    return_sequences=True,
                    name=f'RNNDecoder{i}'
                ), name=f'BiRNNDecoder{i}'
            )
            modeled_ctx = decoder_layer(merged_ctx)

        # 5. 输出层
        span_begin_layer = layers.SpanBegin(name='SpanBegin')
        span_begin_prob = span_begin_layer([merged_ctx, modeled_ctx])

        span_end_layer = layers.SpanEnd(name='SpanEnd')
        span_end_prob = span_end_layer([cencode, merged_ctx, modeled_ctx, span_begin_prob])

        output_layer = layers.Combine(name='CombineOutputs')
        out = output_layer([span_begin_prob, span_end_prob])

        inn = [cinn_word, qinn_word, cinn_char, qinn_char]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )

    def conv1d(self, x_emb):
        # cnn char
        pool_output = []
        kernel_sizes = [2, 3, 4]
        # kernel_sizes = [2]
        for kernel_size in kernel_sizes:
            # x_emb (None, 100, 300)
            # kernel_sizes=2 (None, 20, 300)
            c = self.conv1d_mapper[kernel_size](x_emb)
            # shape=(None, 19, 2), dtype=float32)
            # (none, 1, 2)
            p = self.pool_mapper[kernel_size](c)

            # c = Conv1D(filters=2, kernel_size=kernel_size, strides=1)(x_emb)
            # p = MaxPool1D(pool_size=int(c.shape[1]))(c)
            pool_output.append(p)
            logging.info("kernel_size: %s \t c.shape: %s \t p.shape: %s" % (kernel_size, str(c.shape), str(p.shape)))
        # (None, 1, 6)
        pool_output = concatenate([p for p in pool_output])
        logging.info("pool_output.shape: %s" % str(pool_output.shape))  # (?, 1, 6)
        return pool_output

    def multi_conv1d(self, x_emb):
        # (None, 30, 10, 50)
        words_emb = tf.unstack(x_emb, axis=1)
        vec_list = []
        for word_emb in words_emb:
            vec_list.append(self.conv1d(word_emb))
        char_emb = tf.convert_to_tensor(vec_list)
        # (30, None, 1, 6)
        char_emb = tf.transpose(char_emb, perm=[1, 0, 2, 3])
        # (None,30,1, 6)
        char_emb = tf.squeeze(char_emb, axis=2)
        # (None,30, 6)
        return char_emb


def negative_avg_log_error(y_true, y_pred):
    """
    损失函数计算
    -1/N{sum(i~N)[(log(p1)+log(p2))]}
    :param y_true:
    :param y_pred:
    :return:
    """

    def sum_of_log_prob(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        begin_prob = y_pred_start[begin_idx]
        end_prob = y_pred_end[end_idx]

        return tf.math.log(begin_prob) + tf.math.log(end_prob)

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    batch_prob_sum = tf.map_fn(sum_of_log_prob, inputs, dtype=tf.float32)

    return -tf.keras.backend.mean(batch_prob_sum, axis=0, keepdims=True)


def accuracy(y_true, y_pred):
    """
    准确率计算
    :param y_true:
    :param y_pred:
    :return:
    """

    def calc_acc(inputs):
        y_true, y_pred_start, y_pred_end = inputs

        begin_idx = tf.dtypes.cast(y_true[0], dtype=tf.int32)
        end_idx = tf.dtypes.cast(y_true[1], dtype=tf.int32)

        start_probability = y_pred_start[begin_idx]
        end_probability = y_pred_end[end_idx]

        return (start_probability + end_probability) / 2.0

    y_true = tf.squeeze(y_true)
    y_pred_start = y_pred[:, 0, :]
    y_pred_end = y_pred[:, 1, :]

    inputs = (y_true, y_pred_start, y_pred_end)
    acc = tf.map_fn(calc_acc, inputs, dtype=tf.float32)

    return tf.math.reduce_mean(acc, axis=0)


if __name__ == '__main__':
    ds = preprocess.Preprocessor([
        # './data/squad/train-v1.1.json',
        # './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])

    # train_c, train_q, train_y = ds.get_dataset('./data/squad/train-v1.1.json')
    train_content_word, train_question_word, train_content_char, train_question_char, train_y = ds.get_dataset(
        './data/squad/dev-v1.1.json', pickle_data_path=train_data_pickle_data_path)
    test_content_word, test_question_word, test_content_char, test_question_char, test_y = ds.get_dataset(
        './data/squad/dev-v1.1.json', pickle_data_path=test_data_pickle_data_path)

    print('dataset load done!')

    embedding_matrix_file = './data/glove/embedding_matrix'

    print(len(ds.wordset))
    print(train_content_word.shape, train_content_char.shape, train_y.shape)
    print(test_content_word.shape, test_content_char.shape, test_y.shape)

    embedding_matrix = build_embedding_matrix(ds.word2id, embed_dim=50, embedding_matrix_file=embedding_matrix_file)

    print('embedding_matrix shape {}'.format(embedding_matrix.shape))
    print('embedding done!')

    bidaf = BiDAF(
        clen=ds.max_clen,
        qlen=ds.max_qlen,
        glove_weight=embedding_matrix,
        char_vocab_size=len(ds.char2id)
    )
    bidaf.build_model()

    bidaf.model.fit(
        [train_content_word, train_question_word, train_content_char, train_question_char],
        train_y,
        batch_size=8,
        epochs=10,
        validation_data=([test_content_word, test_question_word, test_content_char, test_question_char], test_y)
    )
