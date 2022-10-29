import math

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class TimeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dense_layer = nn.ModuleDict()
        dense_layer_name = self.get_dense_name()
        for i in dense_layer_name:
            if i[0] == 'h':
                self.dense_layer[i] = nn.Linear(hidden_size,
                                                hidden_size,
                                                bias=False)
            elif i[0] == 't':
                self.dense_layer[i] = nn.Linear(1, hidden_size, bias=False)
            elif i[0] == 'c':
                self.dense_layer[i] = nn.Linear(hidden_size,
                                                hidden_size,
                                                bias=True)
            else:
                self.dense_layer[i] = nn.Linear(input_size,
                                                hidden_size,
                                                bias=True)

    def get_dense_name(self):
        dense_layer_name = ['xf', 'hf', 'xi', 'hi', 'xo', 'ho', 'xz', 'hz']
        dense_layer_name += ['xt', 'ci', 'tt', 't_o', 'cf', 'co']
        return dense_layer_name

    def forward(self, inputs, hc):
        inputs, delta_time = inputs[:, :-1], inputs[:, -1:]
        h, c = hc
        i = torch.sigmoid(self.dense_layer['xi'](inputs) +
                          self.dense_layer['hi'](h) +
                          self.dense_layer['ci'](c))
        f = torch.sigmoid(self.dense_layer['xf'](inputs) +
                          self.dense_layer['hf'](h) +
                          self.dense_layer['cf'](c))
        t = torch.sigmoid(self.dense_layer['xt'](inputs) +
                          torch.sigmoid(self.dense_layer['tt'](delta_time)))
        z = torch.tanh(self.dense_layer['xz'](inputs) +
                       self.dense_layer['hz'](h))

        c = f * c + i * t * z
        o = torch.sigmoid(self.dense_layer['xo'](inputs) +
                          self.dense_layer['t_o'](delta_time) +
                          self.dense_layer['ho'](h) +
                          self.dense_layer['co'](c))
        h = o * torch.tanh(c)
        return h, c


class MotivateCell(TimeLSTMCell):
    def __init__(self, input_size, hidden_size):
        super(MotivateCell, self).__init__(input_size, hidden_size)
        self.intensity_fun = lambda x: 5 / (1 + torch.exp(-x / 5))

    def get_dense_name(self):
        dense_layer_name = [
            'xf', 'hf', 'xi', 'hi', 'xo', 'ho', 'xz', 'hz', 'xt', 'ht'
        ]
        dense_layer_name += ['xf_hat', 'hf_hat', 'xi_hat', 'hi_hat']
        return dense_layer_name

    def forward(self, inputs, state):
        h, c_t, c_hat = state[0], state[1], state[2]
        i = torch.sigmoid(self.dense_layer['xi'](inputs) +
                          self.dense_layer['hi'](h))
        f = torch.sigmoid(self.dense_layer['xf'](inputs) +
                          self.dense_layer['hf'](h))
        o = torch.sigmoid(self.dense_layer['xo'](inputs) +
                          self.dense_layer['ho'](h))
        z = torch.tanh(self.dense_layer['xz'](inputs) +
                       self.dense_layer['hz'](h))
        i_hat = torch.sigmoid(self.dense_layer['xi_hat'](inputs) +
                              self.dense_layer['hi_hat'](h))
        f_hat = torch.sigmoid(self.dense_layer['xf_hat'](inputs) +
                              self.dense_layer['hf_hat'](h))
        t = self.intensity_fun(self.dense_layer['xt'](inputs) +
                               self.dense_layer['ht'](h))

        c = f * c_t + i * z
        c_hat = f_hat * c_hat + i_hat * z
        c_t = c_hat + (c - c_hat) * torch.exp(-t)
        h = o * torch.tanh(c_t)
        return h, c, c_hat


class TimeAwareCell(TimeLSTMCell):
    def __init__(self, input_size, hidden_size, long_time=False):
        super(TimeAwareCell, self).__init__(input_size, hidden_size)
        if long_time:
            self.time_func = lambda delta_t: 1 / torch.log(delta_t + math.e)
        else:
            self.time_func = lambda delta_t: 1 / (delta_t + 1)

    def get_dense_name(self):
        return ['xf', 'hf', 'xi', 'hi', 'xo', 'ho', 'xg', 'hg', 'cd']

    def forward(self, inputs, hc):
        inputs, delta_time = inputs[:, :-1], inputs[:, -1:]
        h, c = hc
        c_short = torch.tanh(self.dense_layer['cd'](c))
        c_short_dis = c_short * self.time_func(delta_time)
        c_long = c - c_short
        c_adjusted = c_long + c_short_dis
        f = torch.sigmoid(self.dense_layer['xf'](inputs) +
                          self.dense_layer['hf'](h))
        i = torch.sigmoid(self.dense_layer['xi'](inputs) +
                          self.dense_layer['hi'](h))
        o = torch.sigmoid(self.dense_layer['xo'](inputs) +
                          self.dense_layer['ho'](h))
        g = torch.tanh(self.dense_layer['xg'](inputs) +
                       self.dense_layer['hg'](h))

        c = f * c_adjusted + i * g
        h = o * torch.tanh(c)
        return h, c


class TimeAwareRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 long_time=False):
        super(TimeAwareRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = TimeAwareCell(input_size, hidden_size, long_time)
        self.v1 = nn.Linear(2 * hidden_size, 1)

    def forward(self, inputs, initial_states=None):
        if isinstance(inputs, rnn_utils.PackedSequence):
            inputs, batch_sizes = inputs.data, inputs.batch_sizes
            inputs = torch.split(inputs, batch_sizes.tolist())
        else:
            if self.batch_first:
                inputs = torch.permute(inputs, (1, 0, 2))  # seq_len,bs,dim
            batch_sizes = [inputs.shape[1]] * inputs.shape[0]
        if initial_states is None:
            zeros = torch.autograd.Variable(
                torch.zeros(
                    (batch_sizes[0], self.hidden_size)).to(inputs[0].device))
            initial_states = (zeros, zeros)
        h, c = initial_states
        length = len(batch_sizes)
        temp_h = None
        for t in range(length):
            h, c = h[:batch_sizes[t]], c[:batch_sizes[t]]
            h, c = self.cell(inputs[t], (h, c))
            if temp_h is None:
                temp_h = torch.unsqueeze(h, dim=0)
            else:
                temp_h = torch.cat(
                    [temp_h[:, :batch_sizes[t], :],
                     torch.unsqueeze(h, dim=0)],
                    dim=0)
        a1 = self.v1(torch.cat([temp_h, temp_h[-1:].repeat(length, 1, 1)],
                               -1))  # seq,bs,1
        a2 = torch.softmax(torch.tanh(a1), dim=0)  # seq,bs,1
        return torch.sum(temp_h * a2, dim=0)


class SimpleRNN(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 cell=None,
                 **kwargs):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.states_num = 2
        if cell is None:
            self.cell = nn.LSTMCell(input_size, hidden_size)
        elif type(cell) is str:
            if cell == 'LSTM':
                self.cell = nn.LSTMCell(input_size, hidden_size)
            elif cell == 'TimeLSTM':
                self.cell = TimeLSTMCell(input_size, hidden_size)
            elif cell == 'Motivate':
                self.cell = MotivateCell(input_size, hidden_size)
                self.states_num = 3
            elif cell == 'TimeAware':
                self.cell = TimeAwareCell(input_size, hidden_size,
                                          kwargs['long_time'])
        else:
            self.cell = cell

    def forward(self, inputs, label_len=4, initial_states=None):
        if isinstance(inputs, rnn_utils.PackedSequence):
            inputs, batch_sizes = inputs.data, inputs.batch_sizes
            inputs = torch.split(inputs, batch_sizes.tolist())
        else:
            if self.batch_first:
                inputs = torch.permute(inputs, (1, 0, 2))  # seq_len,bs,dim
            batch_sizes = torch.tensor([inputs.shape[1]] * inputs.shape[0])
        if initial_states is None:
            zeros = torch.autograd.Variable(
                torch.zeros(
                    (batch_sizes[0], self.hidden_size)).to(inputs[0].device))
            initial_states = [zeros] * self.states_num
        batch_sizes = batch_sizes.to(inputs[0].device)
        states = initial_states
        length = len(batch_sizes)
        temp_h = []
        a1 = []
        a2 = []
        label_where = batch_sizes - torch.cat([
            batch_sizes[label_len:],
            torch.zeros(label_len, dtype=torch.int, device=inputs[0].device)
        ])
        already_label = torch.zeros(batch_sizes[0],
                                    dtype=torch.long,
                                    device=inputs[0].device)
        for t in range(length):
            states = [_[:batch_sizes[t]] for _ in states]
            states = self.cell(inputs[t], states)
            h = states[0]
            if label_where[t] != 0:
                temp_h.append(h[-label_where[t]:])
                a1.append(
                    torch.arange(batch_sizes[t] - label_where[t],
                                 batch_sizes[t]))
                a2.append(
                    torch.clone(
                        already_label[:batch_sizes[t]][-label_where[t]:]))
                already_label[:batch_sizes[t]][-label_where[t]:] += 1
        a1 = torch.cat(a1)
        a2 = torch.cat(a2)
        h = torch.zeros((batch_sizes[0], label_len, self.hidden_size),
                        device=inputs[0].device)
        h[a1, a2] = torch.cat(temp_h)
        return h


class RRN(SimpleRNN):
    # LSTM with time
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 cell=None,
                 **kwargs):
        super(RRN, self).__init__(input_size + 1, hidden_size, batch_first,
                                  cell, **kwargs)


class DUPN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(DUPN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=batch_first)

        self.A1 = nn.Linear(input_size, hidden_size, bias=False)
        self.A2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v1 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, label_len=4):
        h, _ = self.rnn(inputs)
        h = h.data
        inputs, batch_sizes = inputs.data, inputs.batch_sizes
        a1 = torch.sigmoid(self.v1(self.A1(inputs) + self.A2(h)))
        a1 = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(a1, batch_sizes))[0]
        h = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(h, batch_sizes))[0]
        batch_sizes = batch_sizes.to(inputs.device)
        length = len(batch_sizes)
        temp_h = []
        l1 = []
        l2 = []
        label_where = batch_sizes - torch.cat([
            batch_sizes[label_len:],
            torch.zeros(label_len, dtype=torch.int, device=inputs[0].device)
        ])
        already_label = torch.zeros(batch_sizes[0],
                                    dtype=torch.long,
                                    device=inputs[0].device)
        for t in range(length):
            if label_where[t] != 0:
                temp_h.append(
                    torch.sum(torch.softmax(a1[:(t + 1), batch_sizes[t] -
                                                         label_where[t]:batch_sizes[t]],
                                            dim=0) *
                              h[:(t + 1), batch_sizes[t] -
                                          label_where[t]:batch_sizes[t]],
                              dim=0))
                l1.append(
                    torch.arange(batch_sizes[t] - label_where[t],
                                 batch_sizes[t]))
                l2.append(
                    torch.clone(
                        already_label[:batch_sizes[t]][-label_where[t]:]))
                already_label[:batch_sizes[t]][-label_where[t]:] += 1
        l1 = torch.cat(l1)
        l2 = torch.cat(l2)
        h = torch.zeros((batch_sizes[0], label_len, self.hidden_size),
                        device=inputs[0].device)
        h[l1, l2] = torch.cat(temp_h)
        return h


class NARM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(NARM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_l = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.gru_g = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.A1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.A2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v1 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, inputs, label_len=4):
        h_g, _ = self.gru_g(inputs)
        h_l, _ = self.gru_l(inputs)
        inputs, batch_sizes = inputs.data, inputs.batch_sizes
        h_l_1 = self.A1(h_l.data)
        h_l_2 = self.A2(h_g.data)
        h_g = rnn_utils.pad_packed_sequence(h_g)[0]
        h_l = rnn_utils.pad_packed_sequence(h_l)[0]
        h_l_1 = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(h_l_1, batch_sizes))[0]
        h_l_2 = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(h_l_2, batch_sizes))[0]
        batch_sizes = batch_sizes.to(inputs.device)
        length = len(batch_sizes)
        temp_h = []
        l1 = []
        l2 = []
        label_where = batch_sizes - torch.cat([
            batch_sizes[label_len:],
            torch.zeros(label_len, dtype=torch.int, device=inputs[0].device)
        ])
        already_label = torch.zeros(batch_sizes[0],
                                    dtype=torch.long,
                                    device=inputs[0].device)
        for t in range(length):
            if label_where[t] != 0:
                temp_h.append(
                    torch.sum(self.v1(
                        torch.sigmoid(h_l_1[:(t + 1), batch_sizes[t] -
                                                      label_where[t]:batch_sizes[t]] +
                                      h_l_2[:(t + 1), batch_sizes[t] -
                                                      label_where[t]:batch_sizes[t]])) *
                              h_l[:(t + 1), batch_sizes[t] -
                                            label_where[t]:batch_sizes[t]],
                              dim=0) +
                    h_g[t, batch_sizes[t] - label_where[t]:batch_sizes[t]])
                l1.append(
                    torch.arange(batch_sizes[t] - label_where[t],
                                 batch_sizes[t]))
                l2.append(
                    torch.clone(
                        already_label[:batch_sizes[t]][-label_where[t]:]))
                already_label[:batch_sizes[t]][-label_where[t]:] += 1
        l1 = torch.cat(l1)
        l2 = torch.cat(l2)
        h = torch.zeros((batch_sizes[0], label_len, self.hidden_size),
                        device=inputs[0].device)
        h[l1, l2] = torch.cat(temp_h)
        return h


class STAMP(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(STAMP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.w0 = nn.Linear(hidden_size, 1, bias=False)
        self.w1 = nn.Linear(input_size, hidden_size, bias=True)
        self.w2 = nn.Linear(input_size, hidden_size, bias=False)
        self.w3 = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, inputs, label_len=4):
        batch_sizes = inputs.batch_sizes
        x1 = self.w1(inputs.data)
        x2 = self.w2(inputs.data)
        inputs = rnn_utils.pad_packed_sequence(inputs)[0]
        m_s = torch.cumsum(inputs, dim=0) / torch.arange(
            1, inputs.shape[0] + 1, device=inputs.device).reshape(-1, 1, 1)
        wms = self.w3(m_s)
        x1 = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(x1, batch_sizes))[0]
        x2 = rnn_utils.pad_packed_sequence(
            rnn_utils.PackedSequence(x2, batch_sizes))[0]
        batch_sizes = batch_sizes.to(inputs.device)
        length = len(batch_sizes)
        temp_h = []
        l1 = []
        l2 = []
        label_where = batch_sizes - torch.cat([
            batch_sizes[label_len:],
            torch.zeros(label_len, dtype=torch.int, device=inputs[0].device)
        ])
        already_label = torch.zeros(batch_sizes[0],
                                    dtype=torch.long,
                                    device=inputs[0].device)
        for t in range(length):
            if label_where[t] != 0:
                temp_h.append(
                    torch.sum(self.w0(
                        torch.sigmoid(x1[:(t + 1), batch_sizes[t] -
                                                   label_where[t]:batch_sizes[t]] +
                                      x2[t:(t + 1), batch_sizes[t] -
                                                    label_where[t]:batch_sizes[t]] +
                                      wms[t:(t + 1), batch_sizes[t] -
                                                     label_where[t]:batch_sizes[t]])) *
                              x1[:(t + 1), batch_sizes[t] -
                                           label_where[t]:batch_sizes[t]],
                              dim=0) +
                    x2[t, batch_sizes[t] - label_where[t]:batch_sizes[t]])
                l1.append(
                    torch.arange(batch_sizes[t] - label_where[t],
                                 batch_sizes[t]))
                l2.append(
                    torch.clone(
                        already_label[:batch_sizes[t]][-label_where[t]:]))
                already_label[:batch_sizes[t]][-label_where[t]:] += 1
        l1 = torch.cat(l1)
        l2 = torch.cat(l2)
        h = torch.zeros((batch_sizes[0], label_len, self.hidden_size),
                        device=inputs[0].device)
        h[l1, l2] = torch.cat(temp_h)
        return h


class ESMM(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 dropout_rate,
                 batch_first=True):
        super(ESMM, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout_rate)
        self.activate = nn.LeakyReLU()
        self.mlp = nn.ModuleDict()
        self.mlp_names = self.get_mlp_names()
        for name in self.mlp_names:
            self.mlp[name] = self.get_mlp(name)

    def get_mlp_names(self):
        return ['ctr', 'cvr']

    def get_mlp(self, name):
        mlp = nn.Sequential(self.dropout,
                            nn.Linear(self.input_size, self.hidden_sizes[0]),
                            self.activate)
        for i in range(1, len(self.hidden_sizes)):
            mlp.add_module('{}_{}_2'.format(name, i), self.dropout)
            mlp.add_module(
                '{}_{}_0'.format(name, i),
                nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            mlp.add_module('{}_{}_1'.format(name, i), self.activate)
        return mlp

    def f1(self, inputs, label_len):
        batch_sizes = inputs.batch_sizes
        inputs = rnn_utils.pad_packed_sequence(inputs)[0]
        batch_sizes = batch_sizes.to(inputs.device)
        length = len(batch_sizes)
        temp_h = []
        l1 = []
        l2 = []
        label_where = batch_sizes - torch.cat([
            batch_sizes[label_len:],
            torch.zeros(label_len, dtype=torch.int, device=inputs[0].device)
        ])
        already_label = torch.zeros(batch_sizes[0],
                                    dtype=torch.long,
                                    device=inputs[0].device)
        for t in range(length):
            if label_where[t] != 0:
                temp_h.append(
                    torch.mean(inputs[:(t + 1), batch_sizes[t] -
                                                label_where[t]:batch_sizes[t]],
                               dim=0))
                l1.append(
                    torch.arange(batch_sizes[t] - label_where[t],
                                 batch_sizes[t]))
                l2.append(
                    torch.clone(
                        already_label[:batch_sizes[t]][-label_where[t]:]))
                already_label[:batch_sizes[t]][-label_where[t]:] += 1
        l1 = torch.cat(l1)
        l2 = torch.cat(l2)
        h = torch.zeros((batch_sizes[0], label_len, self.input_size),
                        device=inputs[0].device)
        h[l1, l2] = torch.cat(temp_h)
        return h

    def forward(self, inputs, label_len):
        h = self.f1(inputs, label_len)
        h_ctr = torch.sigmoid(self.mlp['ctr'](h))
        h_cvr = torch.sigmoid(self.mlp['cvr'](h))
        return h_ctr * h_cvr


class ESMM2(ESMM):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 dropout_rate,
                 batch_first=True):
        super(ESMM2, self).__init__(input_size, hidden_sizes, dropout_rate,
                                    batch_first)

    def get_mlp_names(self):
        return ['ctr', 'cvr', 'car']

    def forward(self, inputs, label_len):
        h = self.f1(inputs, label_len)
        h_ctr = torch.sigmoid(self.mlp['ctr'](h))
        h_car = torch.sigmoid(self.mlp['car'](h))
        h_cvr = torch.sigmoid(self.mlp['cvr'](h))
        return h_ctr * h_car * h_cvr


class MMOE(ESMM):
    def __init__(self,
                 input_size,
                 hidden_sizes,
                 dropout_rate,
                 expert_num=3,
                 batch_first=True):
        super(MMOE, self).__init__(input_size, hidden_sizes, dropout_rate,
                                   batch_first)
        self.expert_num = expert_num
        self.expert_sizes = [input_size, input_size]
        self.softmax = torch.nn.Softmax(dim=-1)
        self.gates = nn.ModuleDict()
        for name in self.mlp_names:
            self.gates[name] = self.get_gate()
        self.experts = nn.ModuleList()
        for i in range(self.expert_num):
            self.experts.append(self.get_expert())

    def get_expert(self):
        expert_layer = nn.Sequential(
            self.dropout, nn.Linear(self.input_size, self.expert_sizes[0]),
            self.activate)
        for i in range(1, len(self.expert_sizes)):
            expert_layer.add_module('{}_{}_2'.format('expert', i),
                                    self.dropout)
            expert_layer.add_module(
                '{}_{}_0'.format('expert', i),
                nn.Linear(self.expert_sizes[i - 1], self.expert_sizes[i]))
            expert_layer.add_module('{}_{}_1'.format('expert', i),
                                    self.activate)
        return expert_layer

    def get_gate(self):
        gate = nn.Sequential(nn.Linear(self.input_size, self.expert_num),
                             self.softmax)
        return gate

    def forward(self, inputs, label_len):
        inputs = self.f1(inputs, label_len)  # (bs,label_len,dim)
        h_expert = []
        for i in range(self.expert_num):
            h_expert.append(self.experts[i](inputs))
        h_expert = torch.stack(h_expert)  # (3,bs,label_len,dim)
        h_expert = torch.permute(h_expert,
                                 [1, 2, 0, 3])  # (bs,label_len,3,dim)
        h_ctr = self.mlp['ctr'](torch.sum(
            h_expert * torch.unsqueeze(self.gates['ctr'](inputs), dim=-1),
            dim=-2))
        h_cvr = self.mlp['cvr'](torch.sum(
            h_expert * torch.unsqueeze(self.gates['cvr'](inputs), dim=-1),
            dim=-2))
        h_ctr = torch.sigmoid(h_ctr)
        h_cvr = torch.sigmoid(h_cvr)
        return h_ctr * h_cvr


class FMLayer(nn.Module):
    def __init__(self, padding_idx=0, n=10, k=5, v=None):
        super(FMLayer, self).__init__()
        self.n = n
        self.k = k
        self.W0 = nn.Parameter(torch.randn(1))
        self.W1 = nn.Embedding(self.n, 1, padding_idx=padding_idx)  # 前两项线性层
        if v is None:
            self.V = nn.Embedding(self.n, self.k,
                                  padding_idx=padding_idx)  # 交互矩阵
        else:
            self.V = v

    def fm_layer(self, x):
        linear_part = torch.sum(torch.squeeze(self.W1(x)), dim=-1) + self.W0
        x = self.V(x)
        
        interaction_part_1 = torch.pow(torch.sum(x, -1), 2)
        interaction_part_2 = torch.sum(torch.pow(x, 2), -1)
        output = linear_part + 0.5 * torch.sum(
            interaction_part_2 - interaction_part_1, -1, keepdim=False)
        
        return output

    def forward(self, x):
        return self.fm_layer(x)
