import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

from BackModels import SimpleRNN, RRN, NARM, STAMP, DUPN
from MyModels import DIN, DIEN


class PredictModel(nn.Module):
    def __init__(self,
                 all_features_num,
                 input_dim,
                 hidden_dim,
                 predict_hidden_width,
                 output_dim=1,
                 dropout_rate=0.5,
                 model='LSTM',
                 with_time=False):
        super(PredictModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.all_features_num = all_features_num
        self.embedding_layer = nn.Embedding(all_features_num + 1,
                                            input_dim,
                                            padding_idx=all_features_num)
        self.with_time = with_time
        if model in ['LSTM', 'TimeLSTM', 'Motivate']:
            self.rnn = SimpleRNN(input_size=input_dim, hidden_size=hidden_dim, cell=model)
        elif model == 'RRN':
            self.rnn = RRN(input_dim, hidden_dim)
        elif model == 'NARM':
            self.rnn = NARM(input_dim, hidden_dim)
        elif model == 'STAMP':
            self.rnn = STAMP(input_dim, hidden_dim)
        elif model == 'DUPN':
            self.rnn = DUPN(input_dim, hidden_dim)
        elif model == 'DIN':
            self.rnn = DIN(input_dim, hidden_dim)
        elif model == 'DIEN':
            self.rnn = DIEN(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.activate = nn.LeakyReLU()
        self.linear_layers = nn.Sequential(
            self.dropout, nn.Linear(hidden_dim, predict_hidden_width[0]),
            self.activate)

        for i in range(1, len(predict_hidden_width)):
            self.linear_layers.add_module('{}_2'.format(i), self.dropout)
            self.linear_layers.add_module(
                '{}_0'.format(i),
                nn.Linear(predict_hidden_width[i - 1],
                          predict_hidden_width[i]))
            self.linear_layers.add_module('{}_1'.format(i), self.activate)
        self.predict_layer = nn.Linear(predict_hidden_width[-1], output_dim)

    def forward(self, x, label_length):
        # x is a PackedSequence
        if not self.with_time:
            x, batch_sizes = x.data, x.batch_sizes
            x = rnn_utils.PackedSequence(
                torch.sum(self.embedding_layer(x), dim=-2), batch_sizes)
        else:
            x, length = rnn_utils.pad_packed_sequence(x)
            x[:, 1:, -1] = torch.diff(x[:, :, -1])
            x[:, 0, -1] = 0
            x = torch.cat([torch.sum(self.embedding_layer(x[:, :, :-1]), dim=-2), x[:, :, -1:]], dim=-1)
            x = rnn_utils.pack_padded_sequence(x, length)
        h = self.rnn(x, label_length)  # (bs,label_length,hidden_dim)
        h = self.linear_layers(h)  # (bs,label_length,dim)
        y_ = self.predict_layer(h)  # (bs,label_length,output_dim)
        return torch.sigmoid(y_)
