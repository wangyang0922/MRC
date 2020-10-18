import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        context_to_query_attention = tf.keras.layers.Softmax(axis=-1)(similarity)
        encoded_question = K.expand_dims(qencode, axis=1)
        c2q = K.sum(K.expand_dims(context_to_query_attention, axis=-1) * encoded_question, -2)
        return c2q

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):

        max_similarity = K.max(similarity, axis=-1)
        c2q = tf.keras.layers.Softmax()(max_similarity)
        c2q = tf.expand_dims(c2q_att, axis=-1)

        weighted_sum = K.sum(K.expand_dims(c2q, axis=-1) * cencode, -2)
        expanded_weighted_sum = K.expand_dims(weighted_sum, 1)
        num_of_repeatations = K.shape(cencode)[1]
        return K.tile(expanded_weighted_sum, [1, num_of_repeatations, 1])