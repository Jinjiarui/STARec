import torch
from torch import nn

from BackModels import TimeAwareRNN, SimpleRNN


class SearchPredictModel(nn.Module):
    def __init__(self, all_users_num, all_features_num, input_dim, hidden_dim, predict_hidden_width, output_dim=1,
                 dropout_rate=0.5, item_num=15, multi_head=4, use_label=False, long_time=False):
        super(SearchPredictModel, self).__init__()
        self.item_num = item_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.all_users_num = all_users_num
        self.all_features_num = all_features_num
        self.use_label = use_label
        self.embedding_layer = nn.Embedding(all_features_num + 1, input_dim, padding_idx=all_features_num)
        search_input, continuous_input = 2 * input_dim, input_dim
        if use_label:
            search_input += 1
            continuous_input += 1
        self.rnn = TimeAwareRNN(input_size=search_input, hidden_size=hidden_dim, batch_first=True, long_time=long_time)
        self.rnn_continuous = SimpleRNN(input_size=continuous_input, hidden_size=hidden_dim, batch_first=True,
                                        cell='TimeAware', long_time=long_time)
        self.v1 = nn.Linear(2 * hidden_dim, multi_head)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear_layers = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim + (2 + multi_head) * hidden_dim, predict_hidden_width[0]),
            nn.LeakyReLU())

        for i in range(1, len(predict_hidden_width)):
            self.linear_layers.add_module('{}_2'.format(i), nn.Dropout(dropout_rate))
            self.linear_layers.add_module('{}_0'.format(i),
                                          nn.Linear(predict_hidden_width[i - 1], predict_hidden_width[i]))
            self.linear_layers.add_module('{}_1'.format(i), nn.LeakyReLU())
        self.predict_layer = nn.Linear(predict_hidden_width[-1], output_dim)

    def return_user_embedding(self):
        return self.embedding_layer[:self.all_users_num]

    def get_top_k(self, x, self_loc):
        real_loc = torch.not_equal(x[:, :, :, 0], self.all_features_num)  # (bs,users+1,max_len)

        x_self = x[list(range(len(self_loc))), 0, self_loc, -3]  # (bs,)
        x_self = torch.unsqueeze(x_self, dim=-1)  # (bs,1,)
        x_self = torch.unsqueeze(x_self, dim=-1)  # (bs,1,1)
        x_self = x_self.repeat([1, x.shape[1], x.shape[2]])[real_loc]  # (real_items,)
        x_self = self.embedding_layer(x_self)  # (real_items,input_size)

        x_real_loc = x[:, :, :, -3][real_loc]  # (real_items,)
        x_real_loc = self.embedding_layer(x_real_loc)  # (real_items,input_size)

        category_sim = torch.zeros_like(real_loc, dtype=torch.float32)
        category_sim[real_loc] = torch.cosine_similarity(x_self, x_real_loc)  # (bs,users+1,max_len)
        # category_sim[real_loc] += x[:, :, :, -1][real_loc] * 0.5
        category_sim[torch.logical_not(real_loc)] = torch.min(category_sim[real_loc]) - 1
        top_item_idx = torch.topk(category_sim, self.item_num, -1)[1].sort(dim=-1)[0]
        a1 = torch.unsqueeze(torch.arange(category_sim.shape[0]), dim=-1) \
            .repeat([1, category_sim.shape[1]]).unsqueeze(-1).repeat([1, 1, self.item_num])
        a2 = torch.unsqueeze(torch.arange(category_sim.shape[1]), dim=0) \
            .repeat([category_sim.shape[0], 1]).unsqueeze(-1).repeat([1, 1, self.item_num])
        x_self_full = x[list(range(len(self_loc))), 0, self_loc]  # (bs,fields+2)
        x = x[a1, a2, top_item_idx]  # (bs,users+1,item_num,fields+2)
        x[:, 0, -1] = x_self_full
        return x

    def get_rnn_input(self, x):
        if not self.use_label:  # user of label trick
            x, x_time_label = x[:, :, :, :-2], x[:, :, :, -2:-1]
        else:
            x, x_time_label = x[:, :, :, :-2], x[:, :, :, -2:]
            x_time_label[:, :, :, -1] = x_time_label[:, :, :, -1].roll(1, -1)
            x_time_label[:, :, 0, -1] = 0
            x_time_label = x_time_label[:, :, :, [1, 0]]
        x_time_label[:, :, 1:, -1] = torch.diff(x_time_label[:, :, :, -1])
        x_time_label[:, :, 0, -1] = 0
        x = torch.sum(self.embedding_layer(x), dim=-2, keepdim=False)  # (bs*label_len,users+1,item_num,input_size)
        x_self = x[:, 0:1, -1:, :]  # (bs*label_len,1,1,input_size)
        x = torch.cat([x, x_time_label], dim=-1)
        return x, x_self

    def forward(self, x, x_continuous, self_loc):
        # x: (bs*label_len,users+1,max_len,fields+2) 2 is time and label
        # x_continuous: (bs,item_num,fields+2)
        batch_size = x.shape[0]
        label_len = int(batch_size / x_continuous.shape[0])

        x = self.get_top_k(x, self_loc)  # (bs*label_len,users+1,item_num,fields+1)
        x, x_self = self.get_rnn_input(x)  # (bs*label_len,user+1,item_num,input_size) (bs*label_len,1,1,input_size)
        x = torch.cat([x_self.repeat([1, x.shape[1], x.shape[2], 1]), x], dim=-1)
        x = torch.reshape(x, [-1, self.item_num, x.shape[-1]])  # (bs*label_len*(users+1),item_num,input_size*2)
        x_self = torch.squeeze(x_self)  # (bs*label_len,input_size)

        x_continuous = torch.unsqueeze(x_continuous, dim=1)  # (bs,item_num,1,fields+2)
        x_continuous, _ = self.get_rnn_input(x_continuous)  # (bs,item_num,1,input_size)
        x_continuous = torch.squeeze(x_continuous)  # (bs,item_num,input_size)

        h = self.rnn(x)  # (bs*label_len*(users+1),hidden)
        h = torch.reshape(h, [batch_size, -1, self.hidden_dim])  # (bs*label_len,users+1,hidden)
        h_continuous = self.rnn_continuous(x_continuous, label_len)  # (bs,label_len,hidden)
        h_continuous = torch.reshape(h_continuous, [batch_size, -1])  # (bs*label_len,hidden)
        a1 = self.v1(torch.cat([h, h[:, :1, :].repeat(1, h.shape[1], 1)], -1))  # bs*label_len,users+1,multi_head
        a2 = torch.softmax(torch.tanh(a1), dim=1)  # (bs*label_len,users+1,multi_head)
        h_attention = torch.cat([torch.sum(a2[:, :, i:i + 1] * h, dim=1) for i in
                                 range(a2.shape[-1])], dim=-1)  # (bs*label_len,multi_head*hidden)

        h = torch.cat([h[:, 0, :], h_attention], dim=-1)  # (bs*label_len,(multi_head+1)*hidden)
        h = torch.cat([x_self, h, h_continuous], dim=-1)  # (bs*label_len,input_size+(multi_head+2)*hidden)
        h = self.linear_layers(h)
        y_ = self.predict_layer(self.dropout(h))  # (bs*label_len,output_dim)
        return torch.sigmoid(y_)
