import xgboost
import pandas
import numpy
from copy import deepcopy
import woe
import matplotlib.pyplot as plt


def main(train_df, test_df):
    # plt.figure(dpi=2400)
    data = pandas.DataFrame(train_df)
    test = pandas.DataFrame(test_df)
    label = deepcopy(pandas.DataFrame(data['Survived']).values)

    del data['Name']
    del data['Ticket']
    del data['Cabin']
    del data['Survived']
    del data['PassengerId']
    del test['Name']
    del test['Ticket']
    del test['Cabin']
    del test['PassengerId']

    fill_missing_age(data)
    fill_missing_age(test)

    encode_sex(data)
    encode_embarked(data)
    encode_sex(test)
    encode_embarked(test)
    feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    dtrain = xgboost.DMatrix(data.to_numpy(), label=label, feature_names=feature_names)
    # , feature_names=feature_names)
    dtest = xgboost.DMatrix(test.to_numpy(), feature_names=feature_names)

    param = {'max_depth': 3, 'eta': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    param['eval_metric'] = ['auc', 'ams@0']
    # evallist = [(dtest, 'eval'), (dtrain, 'train')]

    num_round = 10
    bst = xgboost.train(param, dtrain, num_round)
    # , evallist)
    bst.save_model('0001.model')
    bst.dump_model('dump.raw.txt', 'featmap.txt')

    ypred = bst.predict(dtest)
    print(ypred)
    res_df = pandas.DataFrame(ypred)
    res_df.to_csv("result3.csv")

    xgboost.plot_importance(bst)
    xgboost.plot_tree(bst)
    plt.show()


def fill_missing_age(data):
    for i in data.index:
        if pandas.isna(data.at[i, 'Age']):
            data.at[i, 'Age'] = 30


def encode_embarked(data):
    for i in data.index:
        if data.at[i, 'Embarked'] == 'S':
            data.at[i, 'Embarked'] = 0
        elif data.at[i, 'Embarked'] == 'C':
            data.at[i, 'Embarked'] = 1
        else:
            data.at[i, 'Embarked'] = 2


def encode_sex(data):
    for i in data.index:
        if data.at[i, 'Sex'] == 'male':
            data.at[i, 'Sex'] = 0
        else:
            data.at[i, 'Sex'] = 1

