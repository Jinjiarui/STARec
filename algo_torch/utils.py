import random

import numpy as np
import torch

from SearchPredictModel import SearchPredictModel
from PredictModel import PredictModel
from PredictModel_NotRNN import PredictModel_NotRNN
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print("random seed set to be " + str(seed))


def load_model(args):
    if args['model'] == 'STARec':
        model = SearchPredictModel(all_users_num=args['all_users_num'],
                                   all_features_num=args['all_features_num'],
                                   input_dim=args['input_dim'],
                                   hidden_dim=args['hidden_dim'],
                                   predict_hidden_width=args['predict_hidden_width'],
                                   output_dim=args['output_dim'],
                                   dropout_rate=args['dropout_rate'],
                                   item_num=args['item_num'],
                                   use_label=args['use_label'],
                                   long_time=args['long_time']
                                   )
    elif args['model'] in ['ESMM', 'ESMM2', 'MMOE']:
        model = PredictModel_NotRNN(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_sizes=args['hidden_dim'],
            dropout_rate=args['dropout_rate'],
            model=args['model'])
    else:
        model = PredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            output_dim=args['output_dim'],
            dropout_rate=args['dropout_rate'],
            model=args['model'],
            with_time=args['with_time']
        )

    return model


def evaluate_utils(y_, y, criterion=None):
    if criterion is not None:
        loss = criterion(y_, y)
    else:
        loss = None
    y_ = y_.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    pred_binary = np.ones_like(y_)
    pred_binary[y_ <= 0.1] = 0
    return loss, accuracy_score(y_true=y, y_pred=pred_binary), roc_auc_score(y_true=y, y_score=y_), \
           f1_score(y_true=y, y_pred=pred_binary)
