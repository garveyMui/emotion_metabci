import pytorch_lightning as pl
from DeepPurpose.encoders import *
from torch.utils.tensorboard import SummaryWriter

from convca import *
from deepnet import *
from eegnet import *
from encoders import *
from guney_net import *
from pretraining import *
from shallownet import *


class Classifier(nn.Sequential):
    def __init__(self, model, **config):
        super(Classifier, self).__init__()
        if config['encoder'] in ['convca', 'deepnet', 'eegnet', 'guney_net', 'shallownet']:
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

    def forward(self, v):
        v = self.model(v)

        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                v = l(v)
                v = F.gelu(v)
            else:
                v = F.gelu(self.dropout(l(v)))  # you can add softmax here

        return v


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
        encoder = config['encoder']

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
        else:
            raise AttributeError('Please use one of the available encoding method.')

        self.model = Classifier(self.encoder, **config)

        self.config = config

        if 'cuda_id' in self.config:
            if self.config['cuda_id'] is None:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(
                    'cuda:' + str(self.config['cuda_id']) if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = encoder

        self.result_folder = config['result_folder']
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.predictor(x)
        return x

    def test_(self, data_generator, model, repurposing_mode=False, test=False):
        y_pred = []
        y_label = []
        model.eval()
        for i, (v, label) in enumerate(data_generator):
            score = self.model(v)
            if self.binary:
                m = torch.nn.Sigmoid()
                logits = torch.squeeze(m(score)).detach().cpu().numpy()
            else:
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1)
                loss = loss_fct(n, Variable(torch.from_numpy(np.array(label)).float()).to(self.device))
                logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
        model.train()
        if self.binary:
            if repurposing_mode:
                return y_pred
            ## ROC-AUC curve
            if test:
                roc_auc_file = os.path.join(self.result_folder, "roc-auc.jpg")
                plt.figure(0)
                roc_curve(y_pred, y_label, roc_auc_file, self.encoder)
                plt.figure(1)
                pr_auc_file = os.path.join(self.result_folder, "pr-auc.jpg")
                prauc_curve(y_pred, y_label, pr_auc_file, self.encoder)

            return roc_auc_score(y_label, y_pred), average_precision_score(y_label, y_pred), f1_score(y_label,
                                                                                                      outputs), log_loss(
                y_label, outputs), y_pred
        else:
            if repurposing_mode:
                return y_pred
            return mean_squared_error(y_label, y_pred), pearsonr(y_label, y_pred)[0], pearsonr(y_label, y_pred)[
                1], concordance_index(y_label, y_pred), y_pred, loss

    def train(self, train, val=None, test=None, verbose=True):
        if len(train.Label.unique()) == 2:
            self.binary = True
            self.config['binary'] = True
        elif len(train.Label.unique()) > 2:
            self.multiclass = True

        lr = self.config['LR']
        decay = self.config['decay']
        BATCH_SIZE = self.config['batch_size']
        train_epoch = self.config['train_epoch']
        if 'test_every_X_epoch' in self.config.keys():
            test_every_X_epoch = self.config['test_every_X_epoch']
        else:
            test_every_X_epoch = 40
        loss_history = []

        self.model = self.model.to(self.device)

        # support multiple GPUs
        if torch.cuda.device_count() > 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPUs!")
            self.model = nn.DataParallel(self.model, dim=0)
        elif torch.cuda.device_count() == 1:
            if verbose:
                print("Let's use " + str(torch.cuda.device_count()) + " GPU!")
        else:
            if verbose:
                print("Let's use CPU/s!")

        opt = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=decay)
        if verbose:
            print('--- Data Preparation ---')

        params = {'batch_size': BATCH_SIZE,
                  'shuffle': True,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False}

        training_generator = data.DataLoader(
            data_process_loader(train.index.values, train.Label.values, train, **self.config), **params)
        if val is not None:
            validation_generator = data.DataLoader(
                data_process_loader(val.index.values, val.Label.values, val, **self.config), **params)

        if test is not None:
            info = data_process_loader(test.index.values, test.Label.values, test, **self.config)
            params_test = {'batch_size': BATCH_SIZE,
                           'shuffle': False,
                           'num_workers': self.config['num_workers'],
                           'drop_last': False,
                           'sampler': SequentialSampler(info)}

            testing_generator = data.DataLoader(
                data_process_loader(test.index.values, test.Label.values, test, **self.config), **params_test)

        # early stopping
        if self.binary:
            max_auc = 0
        else:
            max_MSE = 10000
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = ["# epoch"]
        if self.binary:
            valid_metric_header.extend(["AUROC", "AUPRC", "F1"])
        elif self.multiclass:
            valid_metric_header.extend(["F1"]) # here **********************
        else:
            valid_metric_header.extend(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
        table = PrettyTable(valid_metric_header)
        float2str = lambda x: '%0.4f' % x
        if verbose:
            print('--- Go for Training ---')
        writer = SummaryWriter()
        t_start = time()
        iteration_loss = 0
        for epo in range(train_epoch):
            for i, (v, label) in enumerate(training_generator):
                score = self.model(v)
                label = Variable(torch.from_numpy(np.array(label)).float()).to(self.device)

                if self.binary:
                    loss_fct = torch.nn.BCELoss()
                    m = torch.nn.Sigmoid()
                    n = torch.squeeze(m(score), 1)
                    loss = loss_fct(n, label)
                else:
                    loss_fct = torch.nn.MSELoss()
                    n = torch.squeeze(score, 1)
                    loss = loss_fct(n, label)
                loss_history.append(loss.item())
                writer.add_scalar("Loss/train", loss.item(), iteration_loss)
                iteration_loss += 1

                opt.zero_grad()
                loss.backward()
                opt.step()

                if verbose:
                    if (i % 100 == 0):
                        t_now = time()
                        print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + \
                              ' with loss ' + str(loss.cpu().detach().numpy())[:7] + \
                              ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours")
                    ### record total run time

            if val is not None:
                ##### validate, select the best model up to now
                with torch.set_grad_enabled(False):
                    if self.binary:
                        ## binary: ROC-AUC, PR-AUC, F1, cross-entropy loss
                        auc, auprc, f1, loss, logits = self.test_(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str, [auc, auprc, f1]))
                        valid_metric_record.append(lst)
                        if auc > max_auc:
                            model_max = copy.deepcopy(self.model)
                            max_auc = auc
                        if verbose:
                            print('Validation at Epoch ' + str(epo + 1) + ', AUROC: ' + str(auc)[:7] + \
                                  ' , AUPRC: ' + str(auprc)[:7] + ' , F1: ' + str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                                  str(loss)[:7])
                    else:
                        ### regression: MSE, Pearson Correlation, with p-value, Concordance Index
                        mse, r2, p_val, CI, logits, loss_val = self.test_(validation_generator, self.model)
                        lst = ["epoch " + str(epo)] + list(map(float2str, [mse, r2, p_val, CI]))
                        valid_metric_record.append(lst)
                        if mse < max_MSE:
                            model_max = copy.deepcopy(self.model)
                            max_MSE = mse
                        if verbose:
                            print('Validation at Epoch ' + str(epo + 1) + ' with loss:' + str(loss_val.item())[
                                                                                          :7] + ', MSE: ' + str(mse)[
                                                                                                            :7] + ' , Pearson Correlation: ' \
                                  + str(r2)[:7] + ' with p-value: ' + str(
                                f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI)[:7])
                            writer.add_scalar("valid/mse", mse, epo)
                            writer.add_scalar("valid/pearson_correlation", r2, epo)
                            writer.add_scalar("valid/concordance_index", CI, epo)
                            writer.add_scalar("Loss/valid", loss_val.item(), iteration_loss)
                table.add_row(lst)
            else:
                model_max = copy.deepcopy(self.model)

        # load early stopped model
        self.model = model_max

        if val is not None:
            #### after training
            prettytable_file = os.path.join(self.result_folder, "valid_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(table.get_string())

        if test is not None:
            if verbose:
                print('--- Go for Testing ---')
            if self.binary:
                auc, auprc, f1, loss, logits = self.test_(testing_generator, model_max, test=True)
                test_table = PrettyTable(["AUROC", "AUPRC", "F1"])
                test_table.add_row(list(map(float2str, [auc, auprc, f1])))
                if verbose:
                    print('Validation at Epoch ' + str(epo + 1) + ' , AUROC: ' + str(auc)[:7] + \
                          ' , AUPRC: ' + str(auprc)[:7] + ' , F1: ' + str(f1)[:7] + ' , Cross-entropy Loss: ' + \
                          str(loss)[:7])
            else:
                mse, r2, p_val, CI, logits, loss_test = self.test_(testing_generator, model_max)
                test_table = PrettyTable(["MSE", "Pearson Correlation", "with p-value", "Concordance Index"])
                test_table.add_row(list(map(float2str, [mse, r2, p_val, CI])))
                if verbose:
                    print('Testing MSE: ' + str(mse) + ' , Pearson Correlation: ' + str(r2)
                          + ' with p-value: ' + str(f"{p_val:.2E}") + ' , Concordance Index: ' + str(CI))
            np.save(os.path.join(self.result_folder, str(self.drug_encoder) + '_' + str(self.target_encoder)
                                 + '_logits.npy'), np.array(logits))

            ######### learning record ###########

            ### 1. test results
            prettytable_file = os.path.join(self.result_folder, "test_markdowntable.txt")
            with open(prettytable_file, 'w') as fp:
                fp.write(test_table.get_string())

        ### 2. learning curve
        fontsize = 16
        iter_num = list(range(1, len(loss_history) + 1))
        plt.figure(3)
        plt.plot(iter_num, loss_history, "bo-")
        plt.xlabel("iteration", fontsize=fontsize)
        plt.ylabel("loss value", fontsize=fontsize)
        pkl_file = os.path.join(self.result_folder, "loss_curve_iter.pkl")
        with open(pkl_file, 'wb') as pck:
            pickle.dump(loss_history, pck)

        fig_file = os.path.join(self.result_folder, "loss_curve.png")
        plt.savefig(fig_file)
        if verbose:
            print('--- Training Finished ---')
            writer.flush()
            writer.close()

    def predict(self, df_data):
        print('predicting...')
        info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
        self.model.to(self.device)
        params = {'batch_size': self.config['batch_size'],
                  'shuffle': False,
                  'num_workers': self.config['num_workers'],
                  'drop_last': False,
                  'sampler': SequentialSampler(info)}

        generator = data.DataLoader(info, **params)

        score = self.test_(generator, self.model, repurposing_mode=True)
        return score

    def save_model(self, path_dir):
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        torch.save(self.model.state_dict(), path_dir + '/model.pt')
        save_dict(path_dir, self.config)

    def load_model(self, path):
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
        # test