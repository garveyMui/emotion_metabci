import pytorch_lightning as pl
# from DeepPurpose.encoders import *
import DeepPurpose.DTI as models
import torch
from timm import create_model
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from metabci.brainda.algorithms.deep_learning import *
from metabci.brainda.algorithms.deep_learning.encoders.LaBraM.modeling_finetune import *
import yaml
import os

from metabci.brainda.algorithms.deep_learning.encoders.LaBraM.utils import load_state_dict


# from .convca import *
# from .deepnet import *
# from .eegnet import *
# from .encoders import *
# from .guney_net import *
# from .pretraining import *
# from .shallownet import *


class Classifier(nn.Sequential):
    def __init__(self, model, **config):
        super(Classifier, self).__init__()
        if config['encoder'] in ['convca', 'deepnet', 'eegnet', 'guney_net', 'shallownet', 'labram']:
            self.model = model
            self.dropout = nn.Dropout(config.get('dropout', 0.1))
            self.predictor = nn.ModuleList([nn.Identity()])
        else:
            self.input_dim = config['dim_representation']
            self.model = model
            self.dropout = nn.Dropout(config.get('dropout', 0.1))
            self.hidden_dims = config['cls_hidden_dims']
            layer_size = len(self.hidden_dims) + 1
            dims = [self.input_dim] + self.hidden_dims + [config['n_classes']]
            self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)])

    def forward(self, v, **params):
        v = self.model(v)

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v = l(v)
                v = F.gelu(v)
            else:
                v = F.gelu(self.dropout(l(v)))  # you can add softmax here

        return v

    def predict(self, v):
        self.eval()
        y = self(v)
        return y

def model_initialize(**config):
    model = EEG_model(**config)
    return model


def model_pretrained(path_dir=None, model=None):
    if model is not None:
        path_dir = download_pretrained_model(model)
    config = load_dict(path_dir)
    model = EEG_model(**config)
    model.load_pretrained(path_dir + '/model.pt')
    return model


class EEG_model(pl.LightningModule):
    def __init__(self, **config):
        super(EEG_model, self).__init__()
        print(config)
        encoder = config['encoder']
        print(encoder)
        self.encoder_name = encoder
        if encoder == 'convca':
            self.encoder = ConvCA(n_channels=config['n_channels'],
                                  n_samples=config['n_samples'],
                                  n_classes=config['n_classes'])
        elif encoder == 'deepnet':
            self.encoder = Deep4Net(n_channels=config['n_channels'],
                                    n_samples=config['n_samples'],
                                    n_classes=config['n_classes'])
        elif encoder == 'eegnet':
            self.encoder = EEGNet(n_channels=config['n_channels'],
                                  n_samples=config['n_samples'],
                                  n_classes=config['n_classes'])
        elif encoder == "guney_net":
            self.encoder = GuneyNet(n_channels=config['n_channels'],
                                    n_samples=config['n_samples'],
                                    n_classes=config['n_classes'],
                                    n_bands=config['n_bands'])
        elif encoder == "shallownet":
            self.encoder = ShallowNet(n_channels=config['n_channels'],
                                      n_samples=config['n_samples'],
                                      n_classes=config['n_classes'])
        elif encoder == "transformer":
            self.encoder = transformer(**config)
        elif encoder == "CNN":
            self.encoder = CNN(**config)
        elif encoder == "CNN_RNN":
            self.encoder = CNN_RNN(**config)
        elif encoder == "MLP":
            self.encoder = MLP(input_dim=config['input_dim'],
                               output_dim=config['output_dim'],
                               hidden_dims_lst=config['hidden_dims_lst'])
        elif encoder == "labram":
            yaml_path = "E:/PycharmProjects/emotion_metabci/emotion_metabci/metabci/brainda/algorithms/deep_learning/encoders/LaBraM/config.yaml"
            with open(yaml_path, 'r') as file:
                config_from_yaml = yaml.safe_load(file)
            self.encoder = create_model(**config_from_yaml['model_config'])
        else:
            raise AttributeError('Please use one of the available encoding method.')

        self.model = Classifier(self.encoder, **config)

        self.config = config

        # if 'cuda_id' in self.config:
        #     if self.config['cuda_id'] is None:
        #         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #     else:
        #         self.device = torch.device(
        #             'cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
        # else:
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.result_folder = config.get('result_folder', "./result")
        if not os.path.exists(self.result_folder):
            os.mkdir(self.result_folder)
        if config['n_classes'] == 2:
            self.binary = True
            self.multiclass = False
        elif config['n_classes'] > 2:
            self.binary = False
            self.multiclass = True
        else:
            self.binary = False
            self.multiclass = False

        if 'num_workers' not in self.config.keys():
            self.config['num_workers'] = 0
        if 'decay' not in self.config.keys():
            self.config['decay'] = 0

    def forward(self, x, **params):
        # x = torch.tensor(x, dtype=torch.float32)
        x = self.model(x, **params)
        return x

    def load_model(self, path_dir):
        if self.encoder_name == "labram":
            self.load_model_labram(path_dir, self.encoder)
    def load_model_labram(self, path, model):
        checkpoint = torch.load(path, map_location='cpu')
        print("Load ckpt from %s" % path)
        checkpoint_model = None
        for model_key in "model|module".split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        if (checkpoint_model is not None) and ("gzip" != ''):
            all_keys = list(checkpoint_model.keys())
            new_dict = OrderedDict()
            for key in all_keys:
                if key.startswith('student.'):
                    new_dict[key[8:]] = checkpoint_model[key]
                else:
                    pass
            checkpoint_model = new_dict

        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        for key in all_keys:
            if "relative_position_index" in key:
                checkpoint_model.pop(key)

        load_state_dict(model, checkpoint_model, prefix='')

if __name__ == '__main__':
    config = {"encoder": "labram",
              "n_channels": 30,
              "n_samples": 200,
              "n_classes": 3}

    model = model_initialize(**config)
    data = np.random.random((1, 30, 1, 200))
    idx_channels = np.arange(31) # 0 for CLS token
    model((data, idx_channels))