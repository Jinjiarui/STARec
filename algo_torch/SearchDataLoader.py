import time

import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data


class SearchDataset(data.Dataset):
    def __init__(self, user_item, category_users, candidate_user_num=100, assist_user_num=5,
                 mode='train'):
        self.category_users = category_users
        self.user_log = user_item['log']
        self.user_len = user_item['begin_len']
        self.user_log_id = np.array([np.arange(i[0], i[0] + i[1]) for i in self.user_len], dtype=np.ndarray)
        self.fields_num = user_item['fields']
        self.fields_num_beside_category = np.sum(self.fields_num[:-1])
        self.fields_num_sum = sum(self.fields_num)
        self.continuous = None
        self.assist_user_num = assist_user_num
        self.candidate_user_num = candidate_user_num
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
        candidate_users_list = []
        label_list = []
        for i in range(self.label_len - 1, -1, -1):
            log_now = self.user_log[self.user_len[user, 0] + self.user_len[user, 1] + self.item_loc - i]
            category = log_now[-3] - self.fields_num_beside_category
            category_users = np.array(self.category_users[category])
            if len(category_users) >= self.candidate_user_num + 1:
                candidate_users = np.random.choice(category_users, self.candidate_user_num + 1, replace=False)
                candidate_users = np.delete(candidate_users, np.where(candidate_users == user))
                if len(candidate_users) == self.candidate_user_num + 1:
                    candidate_users = np.delete(candidate_users, 0)
            else:
                category_users = np.delete(category_users, np.where(category_users == user))
                candidate_users = np.concatenate([category_users,
                                                  np.random.choice(len(self.user_len),
                                                                   self.candidate_user_num + 1 - len(category_users),
                                                                   replace=False)])

                candidate_users = np.delete(candidate_users, np.where(candidate_users == user))
                if len(candidate_users) == self.candidate_user_num + 1:
                    candidate_users = np.delete(candidate_users, -1)
            candidate_users = candidate_users.astype(int)
            candidate_users_list.append(candidate_users)
            label_list.append(log_now[-1])  # log_now[-1] is the label
        if self.item_loc == -1:
            logs_id = self.user_log_id[user]
        else:
            logs_id = self.user_log_id[user][:(self.item_loc + 1)]
        logs = self.user_log[logs_id][-15:]
        # return np.concatenate([logs[:, :-2], logs[:, -1:]], axis=-1), logs[-self.label_len:, -1]
        return [user] * self.label_len, candidate_users_list, label_list, logs


class SearchCollate:
    def __init__(self, dataset, embedding=None, device='cuda'):
        self.dataset = dataset
        self.embedding = embedding
        self.device = device

    def get_embedding(self, embedding):
        self.embedding = embedding

    def collate_fn(self, data_):
        users = np.array([_[0] for _ in data_]).reshape(-1, 1)  # bs*label_len,1
        candidate_users = np.array([_[1] for _ in data_])  # bs,label_len,can
        candidate_users = candidate_users.reshape(-1, candidate_users.shape[-1])  # bs*label_len,can
        labels = [_[2] for _ in data_]  # bs,label_len
        self_continuous_logs = [_[3] for _ in data_]  # bs,fields+1
        user_similarity = torch.cosine_similarity(
            self.embedding(torch.tensor(users, device=self.device, requires_grad=False)),
            self.embedding(torch.tensor(candidate_users, device=self.device, requires_grad=False)), dim=-1)
        top_user = torch.topk(user_similarity, self.dataset.assist_user_num, -1)[1]
        a1 = np.expand_dims(np.arange(0, len(users)), -1).repeat(self.dataset.assist_user_num, -1)
        top_user = top_user.cpu().numpy()
        top_user = candidate_users[a1, top_user]  # (bs*label,assist_user_num)
        top_user_log_id = self.dataset.user_log_id[top_user]
        if self.dataset.item_loc != -1:
            self_log_id = np.array(
                [self.dataset.user_log_id[users[user]][0][
                 :self.dataset.item_loc + 2 + user % self.dataset.label_len - self.dataset.label_len]
                 for user in range(len(users))], dtype=np.ndarray)
        else:
            self_log_id = np.array([self.dataset.user_log_id[user][0] for user in users],
                                   dtype=np.ndarray)
        self_log_id = np.expand_dims(self_log_id, axis=-1)
        all_logs_id = np.concatenate([self_log_id, top_user_log_id], axis=-1,
                                     )  # (bs*label,1+assist_user_num) dtype:array
        all_logs_id = np.concatenate(all_logs_id.ravel())
        logs = self.dataset.user_log[all_logs_id]
        self_len = self.dataset.user_len[users, 1] + self.dataset.item_loc + 1
        self_len[:, 0] -= list(range(self.dataset.label_len - 1, -1, -1)) * (len(labels))
        all_len = np.concatenate(
            [self_len, self.dataset.user_len[top_user, 1]],
            axis=-1)  # bs*label, (1+assist_user_num)
        all_len = all_len.ravel()
        logs_split = torch.split(torch.tensor(logs, requires_grad=False), all_len.tolist())

        result = rnn_utils.pad_sequence(logs_split, batch_first=True, padding_value=self.dataset.fields_num_sum)
        # (bs*label*(1+assist_user_num),max_len,fields)
        result = torch.reshape(result, (len(users), -1, result.shape[-2], result.shape[-1]))
        # (bs*label,1+assist_user_num,max_len,fields)
        return (result,
                torch.tensor(self_continuous_logs),
                torch.tensor(labels, dtype=torch.float32).reshape(-1),
                (self_len - 1).ravel().tolist())


if __name__ == '__main__':
    import os

    dataset = '../Data/TaoBao/tmall/raw_data'
    train_data = SearchDataset(np.load(os.path.join(dataset, 'user_item.npz')),
                               np.load(os.path.join(dataset, 'category_users.npy'), allow_pickle=True),
                               candidate_user_num=200, assist_user_num=2, mode='train')
    Embedding_layer = torch.nn.Embedding(train_data.fields_num[0], 32).to('cuda')

    collate = SearchCollate(train_data, Embedding_layer, device='cuda')
    train_dataloader = data.DataLoader(train_data, batch_size=100, shuffle=False, num_workers=6, collate_fn=lambda _: _)
    i = 0
    t1_ = t1 = time.perf_counter()
    network_time = 0
    print(train_data.fields_num)
    train_data.change_mode('train')
    for _data in train_dataloader:
        print(len(_data))
        print(_data[1])
        X, X_continuous, Y, self_loc = collate.collate_fn(_data)
        print(X_continuous)
        print(X_continuous.shape)
        input()
        network_time += time.perf_counter() - t1_
        i += 1
        # if i > 20:
        #     break
        t1_ = time.perf_counter()
    print(time.perf_counter() - t1, network_time)
