import xgboost
import pandas
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from app.config import common_config
import numpy as np


def main(train_df, test_df, train_cv=True):
    data = pandas.DataFrame(train_df)
    test = pandas.DataFrame(test_df)
    label = deepcopy(pandas.DataFrame(data['Survived']).values)
    del data['Survived']

    feature_names = data.columns.values.tolist()
    dtrain = xgboost.DMatrix(data.to_numpy(), label=label, feature_names=feature_names)
    dtest = xgboost.DMatrix(test.to_numpy(), feature_names=feature_names)
    num_round = 10

    param = {
        'max_depth': 3,
        'eta': 1,
        'objective': 'binary:logistic',
        'nthread': 4,
        'eval_metric': ['auc', 'ams@0'],
        'min_child_weight': 0.232
    }

    if train_cv:
        learning_rate_range = np.linspace(0.1, 1, 10, endpoint=True)
        child_weight_range = np.linspace(0.01, 7, 700, endpoint=True)
        gamma_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colsample_bytree_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

        param_gridsearch_input = [
            (max_depth, min_child_weight, learning_rate, colsample_bytree, gamma)
            for max_depth in range(2, 5)
            for min_child_weight in child_weight_range
            for learning_rate in learning_rate_range
            for colsample_bytree in colsample_bytree_range
            for gamma in gamma_range
        ]

        best_auc = float("-Inf")

        for max_depth, min_child_weigth, learning_rate, colsample_bytree, gamma in param_gridsearch_input:
            param = {
                'max_depth': max_depth,
                'eta': 1,
                'objective': 'binary:logistic',
                'nthread': 16,
                'eval_metric': ['auc', 'rmse'],
                'min_child_weight': min_child_weigth,
                'learning_rate': learning_rate,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma
            }

            cv_results = xgboost.cv(
                param,
                dtrain,
                num_round,
                metrics=['auc', 'ams@0'],
                early_stopping_rounds=10
            )

            result_aucpr = cv_results['test-auc-mean'].max()
            rmse = cv_results['test-rmse-mean']
            if best_auc < result_aucpr:
                print("Auc = {}, max_depth = {}, min_child_weight = {},"
                      " learning_rate ={}, colsample_bytree = {}, gamma = {}, rmse = {}"
                      .format(result_aucpr, max_depth, min_child_weigth,
                              learning_rate, colsample_bytree, gamma, rmse))
                best_auc = result_aucpr
    else:
        bst = xgboost.train(param, dtrain, num_round)
        bst.save_model('0001.model')
        bst.dump_model('dump.raw.txt', 'featmap.txt')

        ypred = bst.predict(dtest)
        pred = pandas.Series(ypred)
        res_df = pandas.DataFrame()
        res_df['PassengerId'] = common_config.PASSENGER_IDX_LIST
        res_df['Survived'] = np.round(pred, 0).astype('int32')
        res_df.to_csv("result_{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), index=False)

        xgboost.plot_importance(bst, importance_type='total_cover')
        plt.savefig('feat_importance_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), dpi=1800)
        xgboost.plot_tree(bst)
        plt.savefig('tree_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), dpi=1800)
