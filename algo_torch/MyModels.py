# import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        super(Dice, self).__init__()
        assert dim == 2 or dim == 3
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim

        if self.dim == 3:
            self.alpha = torch.nn.Parameter(torch.zeros((num_features, 1)))
        elif self.dim == 2:
            self.alpha = torch.nn.Parameter(torch.zeros((num_features,)))

    def forward(self, x):
        if self.dim == 3:
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)

        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x

        return out


class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 bias,
                 norm=True,
                 dropout_rate=0.5,
                 activation='relu',
                 sigmoid=False,
                 dice_dim=2):
        super(FullyConnectedLayer, self).__init__()
        assert len(hidden_size) >= 1 and len(bias) >= 1
        assert len(bias) == len(hidden_size)
        self.sigmoid = sigmoid

        layers = []
        layers.append(nn.Linear(input_size, hidden_size[0], bias=bias[0]))

        for i, h in enumerate(hidden_size[:-1]):
            if norm:
                layers.append(nn.LayerNorm(hidden_size[i]))

            if activation.lower() == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation.lower() == 'dice':
                assert dice_dim
                layers.append(Dice(hidden_size[i], dim=dice_dim))
            elif activation.lower() == 'prelu':
                layers.append(nn.PReLU())
            else:
                raise NotImplementedError

            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(
                nn.Linear(hidden_size[i], hidden_size[i + 1], bias=bias[i]))

        self.fc = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()

        # weight initialization xavier_normal (or glorot_normal in keras, tf)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.output_layer(self.fc(x)) if self.sigmoid else self.fc(x)


class AttentionSequencePoolingLayer(nn.Module):
    def __init__(self, embedding_dim=4):
        super(AttentionSequencePoolingLayer, self).__init__()
        self.local_att = LocalActivationUnit(hidden_size=[64, 16],
                                             bias=[True, True],
                                             embedding_dim=embedding_dim,
                                             batch_norm=False)

    def forward(self, query_ad, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        # output              : size -> batch_size * 1 * embedding_size

        activation_weight = self.local_att(query_ad, user_behavior)

        output = activation_weight * user_behavior
        # sumpooling: output: b, n-1, c -> b, 1, c
        output = output.sum(dim=1).unsqueeze(1)
        return output


class LocalActivationUnit(nn.Module):
    def __init__(self,
                 hidden_size=None,
                 bias=None,
                 embedding_dim=4,
                 batch_norm=False):
        super(LocalActivationUnit, self).__init__()
        if bias is None:
            bias = [True, True]
        if hidden_size is None:
            hidden_size = [80, 40]
        self.fc1 = FullyConnectedLayer(input_size=4 * embedding_dim,
                                       hidden_size=hidden_size,
                                       bias=bias,
                                       norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

        self.fc2 = FullyConnectedLayer(input_size=hidden_size[-1],
                                       hidden_size=[1],
                                       bias=[True],
                                       norm=batch_norm,
                                       activation='dice',
                                       dice_dim=3)

    def forward(self, query, user_behavior):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size

        user_behavior_len = user_behavior.size(1)
        queries = torch.cat([query for _ in range(user_behavior_len)], dim=1)

        attention_input = torch.cat([
            queries, user_behavior, queries - user_behavior,
                                    queries * user_behavior
        ],
            dim=-1)
        attention_output = self.fc1(attention_input)
        attention_output = self.fc2(attention_output)

        return attention_output


class AUGRUCell(nn.Module):
    """ Effect of GRU with attentional update gate (AUGRU)

        Reference:
        -  Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018.
    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(AUGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # (W_ir|W_iz|W_ih)
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size,
                                                   input_size))
        self.register_parameter('weight_ih', self.weight_ih)
        # (W_hr|W_hz|W_hh)
        self.weight_hh = nn.Parameter(
            torch.Tensor(3 * hidden_size, hidden_size))
        self.register_parameter('weight_hh', self.weight_hh)
        if bias:
            # (b_ir|b_iz|b_ih)
            self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_ih)
            # (b_hr|b_hz|b_hh)
            self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
            self.register_parameter('bias_ih', self.bias_hh)
            for tensor in [self.bias_ih, self.bias_hh]:
                nn.init.zeros_(tensor, )
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, inputs, hx, att_score):
        gi = F.linear(inputs, self.weight_ih, self.bias_ih)
        gh = F.linear(hx, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, 2)
        h_r, h_z, h_n = gh.chunk(3, 2)

        reset_gate = torch.sigmoid(i_r + h_r)
        update_gate = torch.sigmoid(i_z + h_z)
        new_state = torch.tanh(i_n + reset_gate * h_n)

        update_gate = att_score * update_gate
        hy = (1. - update_gate) * hx + update_gate * new_state
        return hy


class DIN(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(DIN, self).__init__()
        self.attn = AttentionSequencePoolingLayer(embedding_dim=input_size)

    def forward(self, inputs, label_len=None, initial_states=None):
        inputs, length = rnn_utils.pad_packed_sequence(inputs)
        # inputs pad_packed_sequence: n, b, c, so we need to transpose to b, n, c
        inputs = inputs.transpose(0, 1)
        return self.forward_sequence(inputs, label_len)

    def forward_sequence(self, inputs, label_len=None):
        b, n, c = inputs.shape
        # user is the first n-1
        # query is the last 1
        # user sequence: b, n-1, c
        user_sequence = inputs[:, :-1, :]
        # query_sequence: b, 1, c
        query_sequence = inputs[:, -1, :].unsqueeze(1)
        user = self.attn(query_sequence, user_sequence)
        output = torch.cat([user, query_sequence], dim=2)
        return output


class DIEN(nn.Module):
    def __init__(self, input_size, hidden_size=None):
        super(DIEN, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=input_size,
                          batch_first=True)
        self.attn = AttentionSequencePoolingLayer(embedding_dim=input_size)
        self.augru = AUGRUCell(input_size, input_size)

    def forward(self, inputs, label_len=None, initial_states=None):
        inputs, length = rnn_utils.pad_packed_sequence(inputs)
        # inputs pad_packed_sequence: n, b, c, so we need to transpose to b, n, c
        inputs = inputs.transpose(0, 1)
        return self.forward_sequence(inputs, label_len)

    def forward_sequence(self, inputs, label_len=None):
        b, n, c = inputs.shape
        # user is the first n-1
        # query is the last 1
        # user sequence: b, n-1, c
        user_sequence = inputs[:, :-1, :]
        # query_sequence: b, 1, c
        query_sequence = inputs[:, -1, :].unsqueeze(1)
        # query_sequence aftrer gru: b, n-1, c
        user_sequence, _ = self.gru(user_sequence)
        # attention, same in DIN
        attn = self.attn(query_sequence, user_sequence)
        # hx is used in augru
        hx = torch.zeros_like(user_sequence, device=user_sequence.device)
        user_sequence = self.augru(user_sequence, hx, attn)
        # pick the last of augru's output
        user = user_sequence[:, -1, :].unsqueeze(1)

        output = torch.cat([user, query_sequence], dim=2)
        return output


if __name__ == "__main__":
    # model = DIEN(input_size=256).cuda()
    # we use batch size 20, total 1000 catagories
    # we have 32 length of user data
    # we set local length as 1
    indices = torch.randn(20, 33, 48).cuda()
    model = DIEN(48, 100).cuda()
    output = model.forward_sequence(indices)
    # model = AUGRUCell(input_size=256, hidden_size=256).cuda()
    # state = torch.randn(20, 30, 256).cuda()
    # attn = torch.randn(20, 1, 256).cuda()
    # hx = torch.zeros(20, 30, 256).cuda()
    # output = model(state, hx, attn)

    print("final output", output.shape)
    # print(output)
