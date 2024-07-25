import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch import nn

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score, f1_score, log_loss
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
import pickle

torch.manual_seed(2)
np.random.seed(3)
import copy
from prettytable import PrettyTable

import os

from DeepPurpose.utils import *
from DeepPurpose.model_helper import Encoder_MultipleLayers, Embeddings

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class transformer(nn.Sequential):
    def __init__(self, encoding, **config):
        super(transformer, self).__init__()
        if encoding == 'drug':
            self.emb = Embeddings(config['input_dim_drug'], config['transformer_emb_size_drug'], 50,
                                  config['transformer_dropout_rate'])
            self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_drug'],
                                                  config['transformer_emb_size_drug'],
                                                  config['transformer_intermediate_size_drug'],
                                                  config['transformer_num_attention_heads_drug'],
                                                  config['transformer_attention_probs_dropout'],
                                                  config['transformer_hidden_dropout_rate'])
        elif encoding == 'protein':
            self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545,
                                  config['transformer_dropout_rate'])
            self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'],
                                                  config['transformer_emb_size_target'],
                                                  config['transformer_intermediate_size_target'],
                                                  config['transformer_num_attention_heads_target'],
                                                  config['transformer_attention_probs_dropout'],
                                                  config['transformer_hidden_dropout_rate'])

    ### parameter v (tuple of length 2) is from utils.drug2emb_encoder
    def forward(self, v):
        e = v[0].long().to(device)
        e_mask = v[1].long().to(device)
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        emb = self.emb(e)
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
        return encoded_layers[:, 0]

class CNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN, self).__init__()
        if encoding == 'drug':
            in_ch = [63] + config['cnn_drug_filters']
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))
            # n_size_d = 1000
            self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

        if encoding == 'protein':
            in_ch = [26] + config['cnn_target_filters']
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))

            self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x

    def forward(self, v):
        v = self._forward_features(v.double())
        v = v.view(v.size(0), -1)
        v = self.fc1(v.float())
        return v

class CNN_RNN(nn.Sequential):
    def __init__(self, encoding, **config):
        super(CNN_RNN, self).__init__()
        if encoding == 'drug':
            in_ch = [63] + config['cnn_drug_filters']
            self.in_ch = in_ch[-1]
            kernels = config['cnn_drug_kernels']
            layer_size = len(config['cnn_drug_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_d = self._get_conv_output((63, 100))  # auto get the seq_len of CNN output

            if config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
                self.rnn = nn.LSTM(input_size=in_ch[-1],
                                   hidden_size=config['rnn_drug_hid_dim'],
                                   num_layers=config['rnn_drug_n_layers'],
                                   batch_first=True,
                                   bidirectional=config['rnn_drug_bidirectional'])

            elif config['rnn_Use_GRU_LSTM_drug'] == 'GRU':
                self.rnn = nn.GRU(input_size=in_ch[-1],
                                  hidden_size=config['rnn_drug_hid_dim'],
                                  num_layers=config['rnn_drug_n_layers'],
                                  batch_first=True,
                                  bidirectional=config['rnn_drug_bidirectional'])
            else:
                raise AttributeError('Please use LSTM or GRU.')
            direction = 2 if config['rnn_drug_bidirectional'] else 1
            self.rnn = self.rnn.double()
            self.fc1 = nn.Linear(config['rnn_drug_hid_dim'] * direction * n_size_d, config['hidden_dim_drug'])

        if encoding == 'protein':
            in_ch = [26] + config['cnn_target_filters']
            self.in_ch = in_ch[-1]
            kernels = config['cnn_target_kernels']
            layer_size = len(config['cnn_target_filters'])
            self.conv = nn.ModuleList([nn.Conv1d(in_channels=in_ch[i],
                                                 out_channels=in_ch[i + 1],
                                                 kernel_size=kernels[i]) for i in range(layer_size)])
            self.conv = self.conv.double()
            n_size_p = self._get_conv_output((26, 1000))

            if config['rnn_Use_GRU_LSTM_target'] == 'LSTM':
                self.rnn = nn.LSTM(input_size=in_ch[-1],
                                   hidden_size=config['rnn_target_hid_dim'],
                                   num_layers=config['rnn_target_n_layers'],
                                   batch_first=True,
                                   bidirectional=config['rnn_target_bidirectional'])

            elif config['rnn_Use_GRU_LSTM_target'] == 'GRU':
                self.rnn = nn.GRU(input_size=in_ch[-1],
                                  hidden_size=config['rnn_target_hid_dim'],
                                  num_layers=config['rnn_target_n_layers'],
                                  batch_first=True,
                                  bidirectional=config['rnn_target_bidirectional'])
            else:
                raise AttributeError('Please use LSTM or GRU.')
            direction = 2 if config['rnn_target_bidirectional'] else 1
            self.rnn = self.rnn.double()
            self.fc1 = nn.Linear(config['rnn_target_hid_dim'] * direction * n_size_p, config['hidden_dim_protein'])
        self.encoding = encoding
        self.config = config

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input.double())
        n_size = output_feat.data.view(bs, self.in_ch, -1).size(2)
        return n_size

    def _forward_features(self, x):
        for l in self.conv:
            x = F.relu(l(x))
        return x

    def forward(self, v):
        for l in self.conv:
            v = F.relu(l(v.double()))
        batch_size = v.size(0)
        v = v.view(v.size(0), v.size(2), -1)

        if self.encoding == 'protein':
            if self.config['rnn_Use_GRU_LSTM_target'] == 'LSTM':
                direction = 2 if self.config['rnn_target_bidirectional'] else 1
                h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size,
                                 self.config['rnn_target_hid_dim']).to(device)
                c0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size,
                                 self.config['rnn_target_hid_dim']).to(device)
                v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
            else:
                # GRU
                direction = 2 if self.config['rnn_target_bidirectional'] else 1
                h0 = torch.randn(self.config['rnn_target_n_layers'] * direction, batch_size,
                                 self.config['rnn_target_hid_dim']).to(device)
                v, hn = self.rnn(v.double(), h0.double())
        else:
            if self.config['rnn_Use_GRU_LSTM_drug'] == 'LSTM':
                direction = 2 if self.config['rnn_drug_bidirectional'] else 1
                h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size,
                                 self.config['rnn_drug_hid_dim']).to(device)
                c0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size,
                                 self.config['rnn_drug_hid_dim']).to(device)
                v, (hn, cn) = self.rnn(v.double(), (h0.double(), c0.double()))
            else:
                # GRU
                direction = 2 if self.config['rnn_drug_bidirectional'] else 1
                h0 = torch.randn(self.config['rnn_drug_n_layers'] * direction, batch_size,
                                 self.config['rnn_drug_hid_dim']).to(device)
                v, hn = self.rnn(v.double(), h0.double())
        v = torch.flatten(v, 1)
        v = self.fc1(v.float())
        return v

class MLP(nn.Sequential):
	def __init__(self, input_dim, output_dim, hidden_dims_lst):
		'''
			input_dim (int)
			output_dim (int)
			hidden_dims_lst (list, each element is a integer, indicating the hidden size)

		'''
		super(MLP, self).__init__()
		layer_size = len(hidden_dims_lst) + 1
		dims = [input_dim] + hidden_dims_lst + [output_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		# predict
		v = v.float().to(device)
		for i, l in enumerate(self.predictor):
			v = F.relu(l(v))
		return v