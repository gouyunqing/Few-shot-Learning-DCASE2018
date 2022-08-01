import os
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn


def euclidean_dist_similarity(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return -torch.pow(x - y, 2).sum(2)  # N*M


def cosine_similarity(x, y):
    # x: N x D
    # y: M x D
    cos = nn.CosineSimilarity(dim=0)
    cos_sim = []
    for xi in x:
        cos_sim_i = []
        for yj in y:
            cos_sim_i.append(cos(xi, yj))
        cos_sim_i = torch.stack(cos_sim_i)
        cos_sim.append(cos_sim_i)
    cos_sim = torch.stack(cos_sim)
    return cos_sim  # (N, M)


class Down2d(nn.Module):
    """docstring for Down2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 = x1 * torch.sigmoid(x2)

        return x3


class Up2d(nn.Module):
    """docstring for Up2d."""

    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 = x1 * torch.sigmoid(x2)

        return x3

class Discriminator(nn.Module):
    """docstring for Discriminator."""

    def __init__(self,num_class):
        super(Discriminator, self).__init__()
        self.nl = num_class
        self.d1 = Down2d(1+self.nl, 32, (3, 9), (1, 1), (1, 4))
        self.d2 = Down2d(32+self.nl, 32, (3, 8), (1, 2), (1, 3))
        self.d3 = Down2d(32+self.nl, 32, (3, 8), (1, 2), (1, 3))
        self.d4 = Down2d(32+self.nl, 32, (3, 6), (1, 2), (1, 2))

        self.conv = nn.Conv2d(32+self.nl, 1, (36, 5), (36, 1), (0, 2))
        self.pool = nn.AvgPool2d((1, 5))

    def forward(self, x, c):
        assert c.shape[-1]  == self.nl
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)

        x = self.pool(x)
        x = torch.squeeze(x)
        x = torch.tanh(x)
        return x


class Average_Weighted_Attention(nn.Module):
    def __init__(self, vector_size):
        super(Average_Weighted_Attention, self).__init__()
        self.vector_size = vector_size
        self.weights = nn.Parameter(torch.randn(self.vector_size, 1, requires_grad=True)/np.sqrt(self.vector_size),
                                    requires_grad=True)
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    def forward(self, x):
        original_sizes = x.size()
        x = x.contiguous().view(original_sizes[0]*original_sizes[1], -1)
        x_dot_w = x.mm(self.weights)
        x_dot_w = x_dot_w.view(original_sizes[0], original_sizes[1])
        softmax = nn.Softmax(dim=1)
        alphas = softmax(x_dot_w)
        alphas = alphas.view(-1, 1)
        x = x.mul(alphas)
        x = x.view(original_sizes)
        x = torch.sum(x, dim=1)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, bi=False,
                 device='cuda'):
        """
        NOTE: input size must be directly divisible by 4
        Is also used for speaker classifier
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size # == n_mels/ feats
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_outchannels = 32
        self.m_factor = 2 if bi else 1

        self.device = device  # Legacy now, never actually used

        kernel = 7
        padding = int((kernel-1)/2)
        self.conv1 = nn.Conv2d(1, 16, kernel, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 24, kernel, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(24, self.num_outchannels, kernel, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.lstm1 = nn.LSTM(input_size=self.num_outchannels*(self.input_size//8),
                             hidden_size=self.hidden_size, num_layers=self.num_layers,
                             batch_first=True, bidirectional=bi)
        self.att = Average_Weighted_Attention(self.hidden_size*self.m_factor)

        self.fc = nn.Linear(self.m_factor*hidden_size, 128)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, self.num_classes)

    def forward(self, x_data):

        batch_size = x_data.size(0)
        no_features = x_data.size(3)
        curr_device = x_data.device

        # Convolutional layers
        x_data = self.maxpool1(F.relu(self.conv1(x_data)))
        x_data = self.maxpool2(F.relu(self.conv2(x_data)))
        x_data = self.maxpool3(F.relu(self.conv3(x_data)))
        #x_lens = x_lens//8    # seq_len have got ~4 times shorted
        # x = (B, channels, max_l//4, n_mels//4)

        # Recurrent layers
        x_data = x_data.permute(0,2,1,3)
        x_data = x_data.contiguous().view(batch_size, -1, self.num_outchannels*(no_features//8))
        # Now x = (B, max_l//8, channels*(n_mels//8))

        # x_data = nn.utils.rnn.pack_padded_sequence(x_data, x_lens,
        #                                            batch_first=True,
        #                                            enforce_sorted=True)

        h0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        c0 = torch.zeros(self.m_factor*self.num_layers, batch_size,
                         self.hidden_size).to(device=curr_device, dtype=torch.float)

        # LSTM returns: (seq_len, batch, num_directions * hidden_size),
        #               ((num_layers * num_directions, batch, hidden_size), c_n)
        x_data, _ = self.lstm1(x_data, (h0, c0))

        #x_data, x_lens = torch.nn.utils.rnn.pad_packed_sequence(x_data, batch_first=True)

        x_data = self.att(x_data)

        # Alternate non-attention based method: take the final hidden layer for each sequence
        # x_data = torch.stack([row[x_lens[i]-1] for (i,row) in enumerate(x_data)]) #(B, m_factor*hidden_size)

        embedding = self.drop(F.relu(self.fc(x_data)))

        return embedding  # (B, 64)


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, config, bi=False,
                 device='cuda'):
        super(PrototypicalNetwork, self).__init__()
        self.config = config
        self.encoder = Encoder(input_size, hidden_size, num_layers, num_classes, bi=bi, device=device)

    def forward(self, x_support, x_query):
        '''
        Args:
            x_support: (k=batch_size, n=2, 80, 1024)
            x_query: (k=batch_size, n=2, 80, 1024)

        Returns:(k, n)
        '''
        x_proto = torch.zeros((self.config.class_num, self.config.batchsize_train, 128), dtype=float)  # [n=2, k=batch_size, 64]
        x_q = torch.zeros((self.config.class_num, self.config.batchsize_train, 128), dtype=float)  # [n=2, q=batch_size, 64]
        for i in range(x_support.shape[1]):
            x_support_ith_class = x_support[:, i, :, :].unsqueeze(1)  # [k=batch_size, 1, 80, 1024]
            x_query_ith_class = x_query[:, i, :, :].unsqueeze(1)  # [k=batch_size, 1, 80, 1024]
            x_proto_ith_class = self.encoder(x_support_ith_class)  # [k=batch_size, 64]
            x_q_ith_class = self.encoder(x_query_ith_class)
            x_proto[i, :, :] = x_proto_ith_class
            x_q[i, :, :] = x_q_ith_class

        x_proto = x_proto.mean(1)  # [n, 64]
        x_q = x_q.view(-1, x_q.shape[-1])  # [n*(q=batch_size), 64]

        sim_result = self.similarity(x_q, x_proto)  # (n*q, n)

        log_p_y = F.log_softmax(sim_result, dim=1)

        return log_p_y  # (n*q, n)

    @staticmethod
    def similarity(a, b, sim_type='cosine'):
        methods = {'euclidean': euclidean_dist_similarity, 'cosine': cosine_similarity}
        assert sim_type in methods.keys(), f'type must be in {methods.keys()}'
        return methods[sim_type](a, b)  # 值越大相似度越高

