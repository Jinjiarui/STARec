import tensorflow as tf
import math


class TimeAwareCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(self, hidden_size, long_time=False, **kwargs):
        super(TimeAwareCell, self).__init__(num_units=hidden_size, state_is_tuple=True, **kwargs)
        self.dense_layer = {}
        self.hidden_size = hidden_size
        if long_time:
            self.time_func = lambda delta_t: 1 / tf.log(tf.abs(delta_t) + math.e)
        else:
            self.time_func = lambda delta_t: 1 / (delta_t + 1)

    @property
    def output_size(self):
        return self._num_units

    def get_dense_name(self):
        return ['xf', 'hf', 'xi', 'hi', 'xo', 'ho', 'xg', 'hg', 'cd']

    def build(self, input_shape):
        dense_layer_name = self.get_dense_name()
        for i in dense_layer_name:
            if i[0] == 'x' or i[0] == 'c':
                self.dense_layer[i] = tf.layers.Dense(units=self.hidden_size,
                                                      use_bias=True,
                                                      kernel_initializer='random_normal', name=i)
            else:
                self.dense_layer[i] = tf.layers.Dense(units=self.hidden_size,
                                                      use_bias=False,
                                                      kernel_initializer='random_normal', name=i)
        self.built = True

    def call(self, inputs, hc):
        inputs, delta_time = inputs[:, :-1], inputs[:, -1:]
        h, c = hc
        c_short = tf.tanh(self.dense_layer['cd'](c))
        c_short_dis = c_short * self.time_func(delta_time)
        c_long = c - c_short
        c_adjusted = c_long + c_short_dis
        f = tf.sigmoid(self.dense_layer['xf'](inputs) + self.dense_layer['hf'](h))
        i = tf.sigmoid(self.dense_layer['xi'](inputs) + self.dense_layer['hi'](h))
        o = tf.sigmoid(self.dense_layer['xo'](inputs) + self.dense_layer['ho'](h))
        g = tf.tanh(self.dense_layer['xg'](inputs) + self.dense_layer['hg'](h))

        c = f * c_adjusted + i * g
        h = o * tf.tanh(c)
        return h, tf.nn.rnn_cell.LSTMStateTuple(h, c)


class TimeAwareRNN(tf.keras.layers.Layer):
    def __init__(self, hidden_size, long_time=False, **kwargs):
        super(TimeAwareRNN, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.cell = TimeAwareCell(hidden_size, long_time, **kwargs)
        # self.cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, reuse=tf.AUTO_REUSE)
        self.v1 = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        # inputs (bs,seq,input_size)
        if mask is None:
            h, _ = tf.nn.dynamic_rnn(self.cell, inputs, dtype=inputs.dtype)
        else:
            h, _ = tf.nn.dynamic_rnn(self.cell, inputs,
                                     sequence_length=tf.reduce_sum(tf.cast(mask[:, :, 0], tf.int32), axis=-1),
                                     dtype=inputs.dtype)
        # h = self.layer(inputs, mask=mask)  # (bs,seq,hidden_size)
        a1 = self.v1(tf.concat([h, tf.tile(h[:, -1:], [1, tf.shape(h)[1], 1])], -1))  # bs,seq,1
        a1 = tf.exp(a1)
        if mask is not None:
            a1 = tf.where(mask, a1, tf.zeros_like(a1))
        a2 = a1 / tf.reduce_sum(a1, axis=1, keepdims=True)
        return tf.reduce_sum(h * a2, axis=1)


class PredictMLP(tf.keras.layers.Layer):
    def __init__(self, prediction_hidden_width, keep_prob):
        super(PredictMLP, self).__init__()
        self.drop_out = tf.keras.layers.Dropout
        self.activate = tf.keras.layers.LeakyReLU
        self.prediction_mlp = tf.keras.Sequential()
        for width in prediction_hidden_width:
            self.prediction_mlp.add(self.drop_out(1 - keep_prob))
            self.prediction_mlp.add(tf.keras.layers.Dense(units=width, use_bias=True))
            self.prediction_mlp.add(self.activate())

    def call(self, inputs, **kwargs):
        return self.prediction_mlp(inputs)


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, n, k, w1=None, v=None):
        # n is the max feature
        # k is the embedded size
        super(FMLayer, self).__init__()
        self.W0 = tf.Variable(tf.random.normal([1]), name='W0')
        self.W1 = w1  # 前两项线性层
        self.V = v  # 交互矩阵
        if self.W1 is None:
            self.W1 = tf.Variable(tf.random.normal([n, 1]))
        if self.V is None:
            self.V = tf.Variable(tf.random.normal([n, k]))

    def call(self, inputs, **kwargs):
        # inputs:(bs,n),n is the num of fields
        linear_part = tf.reduce_sum(tf.nn.embedding_lookup(self.W1, inputs), axis=-2) + self.W0  # (bs,1)
        x = tf.nn.embedding_lookup(self.V, inputs)  # (bs,n,k)
        interaction_part_1 = tf.pow(tf.reduce_sum(x, -2), 2)  # (bs,k)
        interaction_part_2 = tf.reduce_sum(tf.pow(x, 2), -2)
        product_part = interaction_part_1 - interaction_part_2
        output = linear_part + 0.5 * product_part
        return output


class PNN(tf.keras.layers.Layer):
    def __init__(self, w1, v):
        super(PNN, self).__init__()
        self.fm = FMLayer(0, 0, w1, v)
        self.V = v
        self.bn_1 = tf.keras.layers.BatchNormalization()
        self.bn_2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        embed_inputs = tf.reduce_sum(tf.nn.embedding_lookup(self.V, inputs), axis=-2)
        fm_inputs = self.fm(inputs)
        # embed_sqrt_mean = tf.sqrt(tf.reduce_sum(embed_inputs ** 2, axis=-1, keepdims=True) + 1e-6)
        # fm_sqrt_mean = tf.sqrt(tf.reduce_sum(fm_inputs ** 2, axis=-1, keepdims=True) + 1e-6)
        # fm_inputs = fm_inputs / (fm_sqrt_mean + 1e-8) * embed_sqrt_mean
        fm_inputs = self.bn_1(fm_inputs)
        embed_inputs = self.bn_2(embed_inputs)

        outputs = tf.concat([fm_inputs, embed_inputs], axis=-1)
        return outputs


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.activation_layer = tf.keras.Sequential()
        self.activation_layer.add(tf.keras.layers.Dense(36))
        self.activation_layer.add(tf.keras.layers.LeakyReLU())
        self.activation_layer.add(tf.keras.layers.Dropout(0.5))
        self.activation_layer.add(tf.keras.layers.Dense(1))

    def call(self, inputs, **kwargs):
        user_behavior, target_item = inputs
        mask = kwargs['mask']
        target_item = tf.tile(tf.expand_dims(target_item, -2), [1, tf.shape(user_behavior)[-2], 1])
        activation_weight = self.activation_layer(
            tf.concat([user_behavior, target_item, user_behavior * target_item], -1))
        activation_weight = tf.exp(activation_weight)
        activation_weight = tf.where(mask, activation_weight, tf.zeros_like(activation_weight))
        activation_weight = activation_weight / (tf.reduce_sum(activation_weight, -2, keepdims=True) + 1e-8)
        user_behavior = tf.reduce_sum(user_behavior * activation_weight, -2)
        return user_behavior


class ESMM(tf.keras.layers.Layer):
    def __init__(self, hidden_size, dropout=0.5, **kwargs):
        super(ESMM, self).__init__()
        self.n_hidden = hidden_size
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.A1 = tf.keras.layers.Dense(units=hidden_size, use_bias=True)
        self.A2 = tf.keras.layers.Dense(units=hidden_size, use_bias=True)

    def call(self, inputs, **kwargs):
        mask = kwargs['mask']
        e1 = self.dropout(self.A1(inputs))
        e2 = self.dropout(self.A2(inputs))
        output = e1 * e2
        mask = tf.cast(mask, tf.float32)
        return tf.reduce_sum(output * mask, axis=-2) / (tf.reduce_sum(mask, axis=-2) + 1e-8)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import numpy as np

    x = np.random.choice(200, 600)
    x = tf.convert_to_tensor(x)
    x = tf.reshape(x, (10, 12, 5))
    fm = FMLayer(200, 64)
    pnn = PNN(fm.W1, fm.V)
    y = pnn(x)
    print(y)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(y))
