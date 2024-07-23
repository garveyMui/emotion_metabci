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
	class transformer(nn.Sequential):
		def __init__(self, encoding, **config):
			super(transformer, self).__init__()
			if encoding == 'drug':
				self.emb = Embeddings(config['input_dim_drug'], config['transformer_emb_size_drug'], 50, config['transformer_dropout_rate'])
				self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_drug'],
														config['transformer_emb_size_drug'],
														config['transformer_intermediate_size_drug'],
														config['transformer_num_attention_heads_drug'],
														config['transformer_attention_probs_dropout'],
														config['transformer_hidden_dropout_rate'])
			elif encoding == 'protein':
				self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545, config['transformer_dropout_rate'])
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
			return encoded_layers[:,0]

pass