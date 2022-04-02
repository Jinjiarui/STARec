import time

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data


class SimpleDataset(data.Dataset):
    def __init__(self, user_item, mode='train', with_time=False):
        super(SimpleDataset, self).__init__()
        self.user_log = user_item['log']
        self.user_len = user_item['begin_len']
        self.user_log_id = np.array([np.arange(i[0], i[0] + i[1]) for i in self.user_len], dtype=np.ndarray)
        self.fields_num = user_item['fields']
        self.fields_num_sum = sum(self.fields_num)
        self.with_time = with_time
        self.item_loc_dict = {'train': -3, 'valid': -2, 'test': -1}
        self.label_len_dict = {'train': 1, 'valid': 1, 'test': 1}
        self.item_loc = self.item_loc_dict[mode]
        self.label_len = self.label_len_dict[mode]

    def change_mode(self, mode='train'):
        self.item_loc = self.item_loc_dict[mode]
        self.label_len = self.label_len_dict[mode]

    def __len__(self):
        return self.user_len.shape[0]

    def __getitem__(self, user):
        if self.item_loc == -1:
            logs_id = self.user_log_id[user]
        else:
            logs_id = self.user_log_id[user][:(self.item_loc + 1)]
        logs = self.user_log[logs_id][-30:]
        if self.with_time:
            return logs[:, :-1], logs[-self.label_len:, -1]
        return logs[:, :-2], logs[-self.label_len:, -1]


def collate_fn(data_):
    data_.sort(key=lambda _: len(_[0]), reverse=True)
    logs = [torch.tensor(_[0]) for _ in data_]  # bs,1
    logs_packed = rnn_utils.pack_sequence(logs)
    labels = torch.tensor([_[1] for _ in data_], dtype=torch.float32)  # (bs,label_len)
    return logs_packed, labels
    # return logs_packed


if __name__ == '__main__':
    import os

    dataset = '../Data/TaoBao/tmall/raw_data'
    train_data = SimpleDataset(np.load(os.path.join(dataset, 'user_item.npz')), mode='train')
    # collate = SimpleCollate(device='cuda')
    train_dataloader = data.DataLoader(train_data, batch_size=20,
                                       num_workers=4, collate_fn=collate_fn, shuffle=True)
    i = 0
    t1_ = t1 = time.perf_counter()
    network_time = 0
    print(train_data.fields_num)
    for X, Y in train_dataloader:
        network_time += time.perf_counter() - t1_
        i += 1
        if i > 2:
            break
        s = rnn_utils.pad_packed_sequence(rnn_utils.PackedSequence(X.data, X.batch_sizes))[0]
        print(s, s.shape)
        t1_ = time.perf_counter()
    print(time.perf_counter() - t1, network_time)
