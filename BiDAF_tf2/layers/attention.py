import tensorflow as tf


# c2q 和 q2c 共享相似度矩阵
# 计算对每一个 context word 而言哪些 query words 和它最相关
class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        # 给qencode在索引1增加一个维度
        qencode = tf.expand_dims(qencode, axis=1)

        c2q_att = tf.keras.activations.softmax(similarity, axis=-1)
        c2q_att = tf.expand_dims(c2q_att, axis=-1)
        c2q_att = tf.math.reduce_sum(c2q_att * qencode, -2)

        return c2q_att


# 计算对每一个 query word 而言哪些 context words 和它最相关
class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):
        # 取相关性矩阵每列最大值
        max_similarity = tf.math.reduce_max(similarity, axis=-1)

        # 对以上矩阵进行 softmax 归一化计算 context 向量加权和
        c2q_att = tf.keras.activations.softmax(max_similarity)
        c2q_att = tf.expand_dims(c2q_att, axis=-1)

        # 一个query word可能挑出几个相似度较高context word，
        # 取每个context最相关的query词，但是这个权值用来对所有的context求和
        weighted_sum = tf.math.reduce_sum(c2q_att * cencode, axis=-2)
        weighted_sum = tf.expand_dims(weighted_sum, 1)

        # context长度
        num_repeat = cencode.shape[1]
        # weighted_sum (None, 1, 100)  num_repeat 30
        q2c_att = tf.tile(weighted_sum, [1, num_repeat, 1])
        # q2c_att (None, 30, 100)
        return q2c_att
