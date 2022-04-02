import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

from BackModels import ESMM, ESMM2, MMOE


class PredictModel_NotRNN(nn.Module):
    def __init__(self, all_features_num, input_dim, hidden_sizes, dropout_rate=0.5, model='ESMM'):
        super(PredictModel_NotRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.all_features_num = all_features_num
        self.embedding_layer = nn.Embedding(all_features_num + 1, input_dim, padding_idx=all_features_num)
        if model == 'ESMM':
            self.mlp = ESMM(input_dim, hidden_sizes, dropout_rate)
        elif model == 'ESMM2':
            self.mlp = ESMM2(input_dim, hidden_sizes, dropout_rate)
        elif model == 'MMOE':
            self.mlp = MMOE(input_dim, hidden_sizes, dropout_rate)

    def forward(self, x, label_length):
        # x is a PackedSequence
        x, batch_sizes = x.data, x.batch_sizes
        x = rnn_utils.PackedSequence(torch.sum(self.embedding_layer(x), dim=-2), batch_sizes)
        y_ = self.mlp(x, label_length)  # (bs,label_length,1)
        return y_
