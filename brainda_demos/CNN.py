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

def run():

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	class CNN(nn.Sequential):
		def __init__(self, encoding, **config):
			super(CNN, self).__init__()
			if encoding == 'drug':
				in_ch = [63] + config['cnn_drug_filters']
				kernels = config['cnn_drug_kernels']
				layer_size = len(config['cnn_drug_filters'])
				self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i],
														out_channels = in_ch[i+1],
														kernel_size = kernels[i]) for i in range(layer_size)])
				self.conv = self.conv.double()
				n_size_d = self._get_conv_output((63, 100))
				#n_size_d = 1000
				self.fc1 = nn.Linear(n_size_d, config['hidden_dim_drug'])

			if encoding == 'protein':
				in_ch = [26] + config['cnn_target_filters']
				kernels = config['cnn_target_kernels']
				layer_size = len(config['cnn_target_filters'])
				self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i],
														out_channels = in_ch[i+1],
														kernel_size = kernels[i]) for i in range(layer_size)])
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

pass