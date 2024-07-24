import DeepPurpose.DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from DeepPurpose.encoders import *


class Classifier(nn.Sequential):
    def __init__(self, model_drug, model_protein, **config):
        super(Classifier, self).__init__()
        self.input_dim_drug = config['hidden_dim_drug']
        self.input_dim_protein = config['hidden_dim_protein']

        self.encoding = model_drug
        self.model_protein = model_protein

        self.dropout = nn.Dropout(0.1)

        self.hidden_dims = config['cls_hidden_dims']
        layer_size = len(self.hidden_dims) + 1
        dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]

        self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v_D, v_P):
        # each encoding
        v_D = self.encoding(v_D)
        v_P = self.model_protein(v_P)
        # concatenate and classify
        v_f = torch.cat((v_D, v_P), 1)
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v_f = l(v_f)
            else:
                v_f = F.relu(self.dropout(l(v_f)))
        return v_f


def model_initialize(**config):
    model = EEG_model(**config)
    return model

def model_pretrained(path_dir = None, model = None):
    if model is not None:
        path_dir = download_pretrained_model(model)
    config = load_dict(path_dir)
    model = EEG_model(**config)
    model.load_pretrained(path_dir + '/model.pt')
    return model
class EEG_model():
    def __init__(self, **config):
        encoding = config['encoding']

        if encoding == 'Morgan' or encoding == 'ErG' or encoding == 'Pubchem' or encoding == 'Daylight' or encoding == 'rdkit_2d_normalized' or encoding == 'ESPF':
            # Future : support multiple encoding scheme for static input
            self.encoding = MLP(config['input_dim_drug'], config['hidden_dim_drug'], config['mlp_hidden_dims_drug'])
        elif encoding == 'CNN':
            self.encoding = CNN('drug', **config)
        elif encoding == 'CNN_RNN':
            self.encoding = CNN_RNN('drug', **config)
        elif encoding == 'Transformer':
            self.encoding = transformer('drug', **config)
        elif encoding == "eegnet":
            # to complete relevant code
            self.encoding = EEG_model(**config)
        else:
            raise AttributeError('Please use one of the available encoding method.')


        self.model = Classifier(self.encoding, **config)
        self.config = config

        if 'cuda_id' in self.config:
            if self.config['cuda_id'] is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(
                    'cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoding = encoding
        
        self.result_folder = config['result_folder']
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        self.binary = False
        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0

    def test_(self):
        pass

    def train(self):
        pass

    def predict(self, df_data):
        '''
            utils.data_process_repurpose_virtual_screening
            pd.DataFrame
        '''
        print('predicting...')
        info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
        self.model.to(self.device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}

        if (self.drug_encoding == "MPNN"):
            params['collate_fn'] = mpnn_collate_func
        elif self.drug_encoding in ['DGL_GCN', 'DGL_NeuralFP', 'DGL_GIN_AttrMasking', 'DGL_GIN_ContextPred',
                                    'DGL_AttentiveFP']:
            params['collate_fn'] = dgl_collate_func

        generator = data.DataLoader(info, **params)

        score = self.test_(generator, self.model, repurposing_mode=True)
        return score

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)

    def load_pretrained(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        state_dict = torch.load(path, map_location=torch.device('cpu'))
        # to support training from multi-gpus data-parallel:

        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

        self.binary = self.config['binary']