import numpy as np
from sklearn.metrics import accuracy_score

from SearchPredictModel import SearchPredictModel, SimSearchPredictModel
from AttentionPredictModel import AttentionPredictModel
from PointPredictModel import PointPredictModel
from SimplePredictModel import SimplePredictModel


def load_model(args):
    model = None
    if args['model'] == 'STARec':
        model = SearchPredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            hard_search=args['hard_search'],
            dropout_rate=args['dropout_rate'],
            item_num=args['item_num'],
            use_label=args['use_label'],
            long_time=args['long_time'],
            lr=args['lr'],
            decay_step=args['decay_step'],
            l2_reg=args['l2_reg']
        )
    elif args['model'] == 'Sim':
        model = SimSearchPredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            dropout_rate=args['dropout_rate'],
            item_num=args['item_num'],
            long_time=args['long_time'],
            lr=args['lr'],
            decay_step=args['decay_step'],
            l2_reg=args['l2_reg']
        )
    elif args['model'] == 'light-STARec':
        model = AttentionPredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            dropout_rate=args['dropout_rate'],
            item_num=args['item_num'],
            lr=args['lr'],
            decay_step=args['decay_step'],
            l2_reg=args['l2_reg'],
            long_time=args['long_time']
        )
    elif args['model'] == 'PNN':
        model = PointPredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            dropout_rate=args['dropout_rate'],
            lr=args['lr'],
            decay_step=args['decay_step'],
            l2_reg=args['l2_reg']
        )
    elif args['model'] == 'SimpleSTARec':
        model = SimplePredictModel(
            all_features_num=args['all_features_num'],
            input_dim=args['input_dim'],
            hidden_dim=args['hidden_dim'],
            predict_hidden_width=args['predict_hidden_width'],
            hard_search=args['hard_search'],
            dropout_rate=args['dropout_rate'],
            item_num=args['item_num'],
            use_label=args['use_label'],
            long_time=args['long_time'],
            lr=args['lr'],
            decay_step=args['decay_step'],
            l2_reg=args['l2_reg']
        )
    return model


def evaluate_utils(y_, y):
    pred_binary = np.ones_like(y_)
    pred_binary[y_ <= 0.1] = 0
    return accuracy_score(y_true=y, y_pred=pred_binary)
