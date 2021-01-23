from app import constants as common_const
import pandas


def ordinal_encode_embarked(data_frame):
    for i in data_frame.index:
        if data_frame.at[i, 'Embarked'] == 'S':
            data_frame.at[i, 'Embarked'] = common_const.embarked_ord_encoding['S']
        elif data_frame.at[i, 'Embarked'] == 'C':
            data_frame.at[i, 'Embarked'] = common_const.embarked_ord_encoding['C']
        elif data_frame.at[i, 'Embarked'] == 'Q':
            data_frame.at[i, 'Embarked'] = common_const.embarked_ord_encoding['Q']
        else:
            data_frame.at[i, 'Embarked'] = common_const.embarked_ord_encoding['U']


def ordinal_encode_sex(data_frame):
    for i in data_frame.index:
        if data_frame.at[i, 'Sex'] == 'male':
            data_frame.at[i, 'Sex'] = common_const.sex_ord_encoding['male']
        else:
            data_frame.at[i, 'Sex'] = common_const.sex_ord_encoding['female']


def one_hot_encode(data_frame, col_name):
    data_frame = pandas.concat([data_frame,
                                pandas.get_dummies(data_frame['{}'.format(col_name)],
                                                   prefix='{}'.format(col_name))], axis=1)
    return data_frame


def ordinal_encode_family(data_frame):
    for i in data_frame.index:
        data_frame.at[i, 'FamilyType'] = common_const.family_type_ord_encoding[data_frame.at[i, 'FamilyType']]
