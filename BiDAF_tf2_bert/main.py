import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
import layers
import preprocess
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

print("tf.__version__:", tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class BiDAF:

    def __init__(
            self, clen, qlen, max_char_len, emb_size,
            vocab_size,
            embedding_matrix,
            conv_layers = [],
            max_features=5000,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
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
        self.max_char_len = max_char_len
        self.max_features = max_features
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.conv_layers = conv_layers
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout

    def build_model(self):
        """
        构建模型
        :return:
        """
        # 1 embedding 层
        # 定义字符级的context, question, 词级的context,question的输入
        cemb = tf.keras.layers.Input(shape=(self.word_clen, self.word_emb_size), name='word_context_input')
        qemb = tf.keras.layers.Input(shape=(self.word_qlen, self.word_emb_size), name='word_question_input')


        highway_layer0 = layers.Highway(name='Highway0')
        chighway0 = tf.keras.layers.TimeDistributed(highway_layer0, name='CHighway0')
        qhighway0 = tf.keras.layers.TimeDistributed(highway_layer0, name='QHighway0')
        cemb1 = chighway0(cemb)
        qemb1 = qhighway0(qemb)

        highway_layer1 = layers.Highway(name='Highway1')
        chighway1 = tf.keras.layers.TimeDistributed(highway_layer1, name='CHighway1')
        qhighway1 = tf.keras.layers.TimeDistributed(highway_layer1, name='QHighway1')
        cemb2 = chighway1(cemb1)
        qemb2 = qhighway1(qemb1)

             

        ## 2. 上下文嵌入层
        # 编码器 双向LSTM
        encoder_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                self.word_emb_size,
                recurrent_dropout=self.encoder_dropout,
                return_sequences=True,
                name='RNNEncoder'
            ), name='BiRNNEncoder'
        )

        cencode = encoder_layer(exec('cemb{}'.format(self.num_highway_layers)))  # 编码context
        qencode = encoder_layer(exec('qemb{}'.format(self.num_highway_layers)))  # 编码question

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
                    self.word_emb_size,
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

        # inn = [char_cinn, word_cinn, char_qinn, word_qinn]
        inn = [cemb, qemb]

        self.model = tf.keras.models.Model(inn, out)
        self.model.summary(line_length=128)

        optimizer = tf.keras.optimizers.Adadelta(lr=1e-2)
        self.model.compile(
            optimizer=optimizer,
            loss=negative_avg_log_error,
            metrics=[accuracy]
        )


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
    print('start')
    ds = preprocess.Preprocessor([
        './data/squad/train-v1.1.json',
        './data/squad/dev-v1.1.json',
        './data/squad/dev-v1.1.json'
    ])
    train_c, train_q, train_y = ds.get_dataset('./data/squad/train-v1.1.json', data_type='train')
    test_c, test_q, test_y = ds.get_dataset('./data/squad/dev-v1.1.json', data_type='test')

    print(train_c.shape, train_q.shape,train_y.shape)
    print(test_c.shape, test_q.shape, test_y.shape)
    
    print('middle')
    bidaf = BiDAF(
        word_clen=ds.max_clen,
        word_qlen=ds.max_qlen,
        # char_emb_size=10,
        word_emb_size=768,
        # glove_w2vec_matrix=ds.wv_matrix
        # emb_size=50,
        # max_features=len(ds.charset)
    )
    bidaf.build_model()
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, mode='max')
    print('fit')
    bidaf.model.fit(
        [train_c, train_q], train_y,
        batch_size=2,
        epochs=10,
        callbacks=[early_stopping],
        validation_data=([test_c, test_q], test_y)
    )
    print('save')
    bidaf.model.save_weights('./checkpoints/bidaf_bert')
