import xgboost
import pandas
from datetime import datetime
from copy import deepcopy
import matplotlib.pyplot as plt
from app.config import common_config
from numpy import round


def main(train_df, test_df):
    data = pandas.DataFrame(train_df)
    test = pandas.DataFrame(test_df)
    label = deepcopy(pandas.DataFrame(data['Survived']).values)
    del data['Survived']

    feature_names = data.columns.values.tolist()

    dtrain = xgboost.DMatrix(data.to_numpy(), label=label, feature_names=feature_names)
    dtest = xgboost.DMatrix(test.to_numpy(), feature_names=feature_names)

    param = {
        'max_depth': 2,
        'eta': 1,
        'objective': 'binary:logistic'
    }

    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['eval_metric'] = ['auc', 'ams@0']

    num_round = 10
    bst = xgboost.train(param, dtrain, num_round)
    bst.save_model('0001.model')
    bst.dump_model('dump.raw.txt', 'featmap.txt')

    ypred = bst.predict(dtest)
    pred = pandas.Series(ypred)
    res_df = pandas.DataFrame()
    res_df['PassengerId'] = common_config.PASSENGER_IDX_LIST
    res_df['Survived'] = round(pred, 0).astype('int32')
    res_df.to_csv("result_{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), index=False)

    xgboost.plot_importance(bst)
    xgboost.plot_tree(bst)
    plt.savefig('figure_{}'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')), dpi=1800)
