import math

import tensorflow as tf

from BackModels import PNN, PredictMLP, Attention


def cosine_similarity(x1, x2):
    norm_1 = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=-1) + 1e-8)
    norm_2 = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=-1) + 1e-8)

    product = tf.reduce_sum(x1 * x2, axis=-1)
    cosine = product / (norm_1 * norm_2)
    return cosine


class AttentionPredictModel(tf.keras.layers.Layer):
    def __init__(self, all_features_num, input_dim, hidden_dim, predict_hidden_width,
                 dropout_rate=0.5, item_num=15, use_label=False, lr=1e-2, decay_step=1000, l2_reg=1e-4,
                 long_time=False):
        super(AttentionPredictModel, self).__init__()
        self.item_num = item_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.all_features_num = all_features_num
        self.use_label = use_label
        self.embedding_matrix = tf.Variable(tf.random_normal([all_features_num + 1, input_dim], stddev=0.1))
        self.w1 = tf.Variable(tf.random_normal([all_features_num + 1, 1], stddev=0.1))
        self.pnn = PNN(self.w1, self.embedding_matrix)
        self.W = tf.keras.layers.Dense(self.hidden_dim)
        self.A = Attention()
        self.A_continuous = Attention()
        self.mlp = PredictMLP(predict_hidden_width, dropout_rate)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc = tf.keras.layers.Dense(1)

        self.global_step = tf.Variable(0, trainable=False)
        cov_learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_step, 0.96)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
        self.l2_reg = l2_reg
        if long_time:
            self.time_func = lambda delta_t: 1 / tf.log(tf.abs(delta_t) + math.e)
        else:
            self.time_func = lambda delta_t: 1 / (delta_t + 1)

    def optimize_loss(self, y, y_, training):
        y_ = tf.sigmoid(y_)
        base_loss = tf.losses.log_loss(labels=y, predictions=y_)
        base_loss = tf.reduce_mean(base_loss)
        loss = base_loss
        for v in tf.trainable_variables():
            loss += self.l2_reg * tf.nn.l2_loss(v)
        tf.summary.scalar('base_loss', base_loss)
        tf.summary.scalar('loss', loss)

        threshold = 0.5
        y, y_ = y[:, 0], y_[:, 0]
        one_click = tf.ones_like(y)
        zero_click = tf.zeros_like(y)
        eval_metric_ops = {
            "auc": tf.metrics.auc(y, y_),
            "acc": tf.metrics.accuracy(y, tf.where(y_ >= threshold, one_click, zero_click))
        }
        if not training:
            return base_loss, loss, eval_metric_ops, y_, y
        gvs, v = zip(*self.optimizer.compute_gradients(loss))
        gvs, _ = tf.clip_by_global_norm(gvs, 5.0)
        gvs = zip(gvs, v)
        train_op = self.optimizer.apply_gradients(gvs, global_step=self.global_step)
        return base_loss, loss, eval_metric_ops, y_, y, train_op

    def get_top_k(self, x, self_loc):
        # x:(bs,max_len,fields+2)
        real_loc = tf.not_equal(x[:, :, 0], self.all_features_num)  # (bs,max_len)
        real_loc = tf.where(real_loc)
        x_self = x[:, :, -3]  # (bs,max_len)
        x_self = tf.gather_nd(x_self, tf.stack([tf.range(tf.shape(self_loc)[0]), self_loc], axis=-1))  # (bs,)
        x_self = tf.expand_dims(x_self, -1)  # (bs,1)
        x_self = tf.tile(x_self, [1, tf.shape(x)[1]])  # (bs,max_len)
        x_self = tf.gather_nd(x_self, real_loc)
        x_self = tf.nn.embedding_lookup(self.embedding_matrix, x_self)  # (real_items,input_dim)
        x_real_loc = tf.gather_nd(x[:, :, -3], real_loc)
        x_real_loc = tf.nn.embedding_lookup(self.embedding_matrix, x_real_loc)  # (real_items,input_dim)
        category_sim = cosine_similarity(x_self, x_real_loc)
        self_cumsum = tf.cumsum(self_loc + 1)
        self_cumsum = tf.concat([[0], self_cumsum], axis=0)
        max_len = tf.shape(x)[1]
        category_sim = tf.map_fn(lambda i:
                                 tf.concat([category_sim[self_cumsum[i]:self_cumsum[i + 1] - 1],
                                            tf.fill([max_len + 1 - self_cumsum[i + 1] + self_cumsum[i]], -2.0)],
                                           axis=0),
                                 tf.range(tf.shape(x)[0]), dtype=tf.float32
                                 )
        top_item_idx = tf.nn.top_k(category_sim, self.item_num, False).indices  # (bs,item_num)
        top_item_idx = tf.nn.top_k(top_item_idx, tf.shape(top_item_idx)[1], False).values
        top_item_idx = tf.reverse(top_item_idx, axis=[-1])
        a1 = tf.expand_dims(tf.range(tf.shape(top_item_idx)[0]), -1)
        a1 = tf.tile(a1, [1, self.item_num])
        top_item_idx = tf.stack([a1, top_item_idx], axis=-1)

        x_self_full = tf.gather_nd(x, tf.stack([tf.range(tf.shape(self_loc)[0]), self_loc], axis=-1))  # (bs,fields+2)
        x_self_full = tf.expand_dims(x_self_full, 1)
        x = tf.concat([tf.gather_nd(x, top_item_idx), x_self_full], axis=1)
        return x, (category_sim, x_real_loc, real_loc)

    def get_rnn_input(self, x):
        if not self.use_label:
            x, x_time = x[:, :, :-2], x[:, :, -2]
        else:  # use of label trick
            x, x_time, x_label = x[:, :, :-2], x[:, :, -2], x[:, :, -1]
            x_label = tf.concat([tf.zeros_like(x_label[:, 0:1]), x_label[:, :-1]], axis=-1)
            x_label = tf.expand_dims(x_label, -1)
            x_label = tf.cast(x_label, tf.float32)
        x = self.pnn(x)
        # x = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_matrix, x), axis=-2)  # (bs,item_num,input_size)
        x_time = tf.concat([tf.zeros_like(x_time[:, 0:1]), x_time[:, 1:] - x_time[:, :-1]], axis=-1)
        x_time = tf.expand_dims(x_time, -1)
        x_time = tf.cast(x_time, tf.float32)
        x_time = self.time_func(x_time)
        if self.use_label:
            x = tf.concat([x, x_label], axis=-1)
        # x = tf.concat([x, x_time], axis=-1)
        return x

    def forward(self, inputs):
        # x: (bs,max_len,fields+2) 2 is time and label
        # x_continuous: (bs,item_num,fields+2)
        x, x_continuous, self_loc, y = inputs
        x, _ = self.get_top_k(x, self_loc)  # (bs,item_num,fields+2)
        mask1 = tf.not_equal(x[:, :, 0:1], self.all_features_num)
        x = self.get_rnn_input(x)  # (bs,item_num,input_size), (bs,1,input_size)
        x = self.W(x)
        x_self = x[:, -1]

        mask2 = tf.not_equal(x_continuous[:, :, 0:1], self.all_features_num)
        x_continuous = self.get_rnn_input(x_continuous)  # (bs,item_num,input_size)
        x_continuous = self.W(x_continuous)

        h = self.A([x, x[:, -1]], mask=mask1)  # (bs,hidden)
        h_continuous = self.A_continuous([x_continuous, x_continuous[:, -1]], mask=mask2)

        h = tf.concat([x_self, h, h_continuous], axis=-1)  # (bs,input_size+hidden_size)
        h = self.mlp(h)
        h = self.dropout(h)
        y_ = self.fc(h)
        y = tf.expand_dims(y, -1)
        return y, y_

    def call(self, inputs, training=True):
        if training:
            tf.keras.backend.set_learning_phase(1)
        else:
            tf.keras.backend.set_learning_phase(0)
        y, y_ = self.forward(inputs)
        return self.optimize_loss(y, y_, training)
