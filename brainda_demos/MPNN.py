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
	class MPNN(nn.Sequential):

		def __init__(self, mpnn_hidden_size, mpnn_depth):
			super(MPNN, self).__init__()
			self.mpnn_hidden_size = mpnn_hidden_size
			self.mpnn_depth = mpnn_depth
			from DeepPurpose.chemutils import ATOM_FDIM, BOND_FDIM

			self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
			self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
			self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)



		## utils.smiles2mpnnfeature -> utils.mpnn_collate_func -> utils.mpnn_feature_collate_func -> encoders.MPNN.forward
		def forward(self, feature):
			'''
				fatoms: (x, 39)
				fbonds: (y, 50)
				agraph: (x, 6)
				bgraph: (y, 6)
			'''
			fatoms, fbonds, agraph, bgraph, N_atoms_bond = feature
			N_atoms_scope = []
			##### tensor feature -> matrix feature
			N_a, N_b = 0, 0
			fatoms_lst, fbonds_lst, agraph_lst, bgraph_lst = [],[],[],[]
			for i in range(N_atoms_bond.shape[0]):
				atom_num = int(N_atoms_bond[i][0].item())
				bond_num = int(N_atoms_bond[i][1].item())

				fatoms_lst.append(fatoms[i,:atom_num,:])
				fbonds_lst.append(fbonds[i,:bond_num,:])
				agraph_lst.append(agraph[i,:atom_num,:] + N_a)
				bgraph_lst.append(bgraph[i,:bond_num,:] + N_b)

				N_atoms_scope.append((N_a, atom_num))
				N_a += atom_num
				N_b += bond_num


			fatoms = torch.cat(fatoms_lst, 0)
			fbonds = torch.cat(fbonds_lst, 0)
			agraph = torch.cat(agraph_lst, 0)
			bgraph = torch.cat(bgraph_lst, 0)
			##### tensor feature -> matrix feature


			agraph = agraph.long()
			bgraph = bgraph.long()

			fatoms = create_var(fatoms).to(device)
			fbonds = create_var(fbonds).to(device)
			agraph = create_var(agraph).to(device)
			bgraph = create_var(bgraph).to(device)

			binput = self.W_i(fbonds) #### (y, d1)
			message = F.relu(binput)  #### (y, d1)

			for i in range(self.mpnn_depth - 1):
				nei_message = index_select_ND(message, 0, bgraph)
				nei_message = nei_message.sum(dim=1)
				nei_message = self.W_h(nei_message)
				message = F.relu(binput + nei_message) ### (y,d1)

			nei_message = index_select_ND(message, 0, agraph)
			nei_message = nei_message.sum(dim=1)
			ainput = torch.cat([fatoms, nei_message], dim=1)
			atom_hiddens = F.relu(self.W_o(ainput))
			output = [torch.mean(atom_hiddens.narrow(0, sts,leng), 0) for sts,leng in N_atoms_scope]
			output = torch.stack(output, 0)
			return output

pass