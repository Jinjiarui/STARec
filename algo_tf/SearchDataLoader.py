import numpy as np
import tensorflow as tf


class DataSet:
    def __init__(self, user_item, batch_size, search_len, hard_search=False):
        super(DataSet, self).__init__()
        self.hard_search = hard_search
        self.user_log = user_item['log']
        self.user_len = user_item['begin_len']
        self.user_log_id = np.array([np.arange(i[0], i[0] + i[1]) for i in self.user_len], dtype=np.ndarray)
        self.fields_num = user_item['fields']
        self.fields_sum = len(self.fields_num)
        self.fields_num_beside_category = np.sum(self.fields_num[:-1])
        self.fields_num_sum = int(sum(self.fields_num))
        self.item_loc_dict = {'train': -3, 'valid': -2, 'test': -1}
        self.batch_size = batch_size
        self.search_len = search_len
        self.continuous_len = 30 - self.search_len

    def search(self, seq):
        if self.hard_search:
            seq = seq[seq[:, -3] == seq[-1, -3]]
            if np.shape(seq)[0] <= self.search_len:
                return seq
            choice = np.sort(np.random.choice(np.shape(seq)[0], self.search_len, replace=False))
            return seq[choice]
        return seq

    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        if mode == 'test':
            for i in self.user_log_id:
                yield self.search(self.user_log[i]), self.user_log[i][-self.continuous_len:], \
                      len(self.user_log[i]) + self.item_loc_dict[mode]
        else:
            for i in self.user_log_id:
                yield self.search(self.user_log[i][:self.item_loc_dict[mode] + 1]), \
                      self.user_log[i][
                      -self.continuous_len + self.item_loc_dict[mode] + 1:self.item_loc_dict[mode] + 1], \
                      len(self.user_log[i]) + self.item_loc_dict[mode]

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator, (tf.int32, tf.int32, tf.int32), args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(2 * self.batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([None, self.fields_sum + 2]),
                                           tf.TensorShape([self.continuous_len, self.fields_sum + 2]),
                                           tf.TensorShape([])),  # 表示后者不补全
                                       padding_values=(self.fields_num_sum, self.fields_num_sum, 0)
                                       )
        dataset = dataset.prefetch(1)
        iter = dataset.make_one_shot_iterator()
        x, x_continuous, self_loc = iter.get_next()
        label = tf.gather_nd(x, tf.stack([tf.range(tf.shape(x)[0]), self_loc], -1))[:, -1]
        # label = x_continuous[:, -1, -1]
        return x, x_continuous, self_loc, tf.cast(label, tf.float32)


class PointDataSet(DataSet):
    def __init__(self, user_item, batch_size):
        super(PointDataSet, self).__init__(user_item, batch_size, 1)

    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        item_loc = self.item_loc_dict[mode]
        for i in self.user_log_id:
            yield self.user_log[i[item_loc]]

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator, tf.int32, tf.TensorShape([self.fields_sum + 2]),
                                                 args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(2 * self.batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        iter = dataset.make_one_shot_iterator()
        x = iter.get_next()
        x, label = x[:, :-2], x[:, -1]
        return x, tf.cast(label, tf.float32)


class SimpleDataset(DataSet):
    def __init__(self, user_item, batch_size, search_len, hard_search=False):
        super(SimpleDataset, self).__init__(user_item, batch_size, search_len, hard_search)
        self.fea_select = [0, 1, -3, -2, -1]

    def from_generator(self, mode):
        mode = str(mode, encoding='utf-8')
        item_loc = self.item_loc_dict[mode]
        if mode == 'test':
            for i in self.user_log_id:
                yield self.search(self.user_log[i][:, self.fea_select]), \
                      self.user_log[i][-self.continuous_len:, self.fea_select], \
                      self.user_log[i][item_loc], len(self.user_log[i]) + item_loc
        else:
            for i in self.user_log_id:
                yield self.search(self.user_log[i][:item_loc + 1, self.fea_select]), \
                      self.user_log[i][-self.continuous_len + item_loc + 1:item_loc + 1, self.fea_select], \
                      self.user_log[i][item_loc], len(self.user_log[i]) + item_loc

    def input_fn(self, mode='train', epochs=1, shuffle=True):
        dataset = tf.data.Dataset.from_generator(self.from_generator, (tf.int32, tf.int32, tf.int32, tf.int32),
                                                 args=(mode,))
        if shuffle:
            dataset = dataset.shuffle(2 * self.batch_size)
        dataset = dataset.repeat(epochs)
        dataset = dataset.padded_batch(self.batch_size,
                                       padded_shapes=(
                                           tf.TensorShape([None, 5]),
                                           tf.TensorShape([self.continuous_len, 5]),
                                           tf.TensorShape([self.fields_sum + 2]), tf.TensorShape([])),  # 表示不补全
                                       padding_values=(self.fields_num_sum, self.fields_num_sum, 0, 0)
                                       )
        dataset = dataset.prefetch(1)
        iter = dataset.make_one_shot_iterator()
        x, x_continuous, target_item, self_loc = iter.get_next()
        label = target_item[:, -1]
        return x, x_continuous, target_item[:, :-2], self_loc, tf.cast(label, tf.float32)


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    dataset = '../Data/TaoBao/tmall/raw_data'
    data = SimpleDataset(np.load(os.path.join(dataset, 'user_item.npz')), 2, 15)
    train_data = data.input_fn(mode='test')
    x, x_continuous, target_item, self_loc, label = train_data
    print(x.shape, label.shape)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        for i in range(2):
            print(sess.run([x_continuous, target_item]))
