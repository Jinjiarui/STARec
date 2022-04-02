import tensorflow as tf

from BackModels import PNN, PredictMLP


class PointPredictModel(tf.keras.layers.Layer):
    def __init__(self, all_features_num, input_dim, predict_hidden_width,
                 dropout_rate=0.5, lr=1e-2, decay_step=1000, l2_reg=1e-4,
                 ):
        super(PointPredictModel, self).__init__()
        self.input_dim = input_dim
        embedding_matrix = tf.Variable(tf.random_normal([all_features_num + 1, input_dim], stddev=0.1))
        w1 = tf.Variable(tf.random_normal([all_features_num + 1, 1], stddev=0.1))
        self.pnn = PNN(w1, embedding_matrix)

        self.mlp = PredictMLP(predict_hidden_width, dropout_rate)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activate = tf.keras.layers.LeakyReLU()
        self.fc = tf.keras.layers.Dense(1)

        self.global_step = tf.Variable(0, trainable=False)
        cov_learning_rate = tf.train.exponential_decay(lr, self.global_step, decay_step, 0.96)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=cov_learning_rate)
        self.l2_reg = l2_reg

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

    def forward(self, inputs):
        # x: (bs,max_len,fields+2) 2 is time and label
        x, y = inputs
        h = self.pnn(x)
        h = self.mlp(h)
        h = self.dropout(h)
        h = self.activate(h)
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

