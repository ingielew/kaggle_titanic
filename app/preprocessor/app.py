from . import encoding
from app import constants as common_const
from re import sub
import pandas


def preprocess(data_frame):
    passenger_deck_df = extract_deck(data_frame)
    handle_age(data_frame)
    families = identify_families(data_frame)
    adjust_fare_per_person(data_frame)
    extract_ticket_code(data_frame)
    encoding.ordinal_encode_embarked(data_frame)
    data_frame = encoding.one_hot_encode(data_frame, 'Sex')
    data_frame = encoding.one_hot_encode(data_frame, 'Ticket')
    data_frame['Deck'] = passenger_deck_df['Deck']
    data_frame['FamilyType'] = families['FamilyType']
    remove_redundant_columns(data_frame)
    return data_frame


def remove_redundant_columns(data_frame):
    del data_frame['Name']
    del data_frame['PassengerId']
    del data_frame['Ticket']
    del data_frame['Embarked']
    del data_frame['Cabin']
    del data_frame['Sex']


def handle_age(data_frame):
    average_age = round(data_frame['Age'].mean(skipna=True))

    for i in data_frame.index:
        if pandas.isna(data_frame.at[i, 'Age']):
            age_est = average_age
            if data_frame.at[i, 'SibSp'] > 0:
                possible_sibsp = data_frame.loc[data_frame['Ticket'] == data_frame.at[i, 'Ticket']]
                avg_sibsp_age = 0
                count = 0
                for sibsp in possible_sibsp.index:
                    if possible_sibsp.at[sibsp, 'PassengerId'] != data_frame.at[i, 'PassengerId']:
                        if not pandas.isna(possible_sibsp.at[sibsp, 'Age']):
                            avg_sibsp_age = avg_sibsp_age + possible_sibsp.at[sibsp, 'Age']
                            count = count + 1
                if count != 0:
                    age_est = avg_sibsp_age/count
            data_frame.at[i, 'Age'] = age_est


def identify_families(data_frame):
    family_df = pandas.DataFrame()
    passenger_record = {
        'PassengerId': 0,
        'FamilyType': 0
    }

    for i in data_frame.index:
        sibsp = data_frame.at[i, 'SibSp']
        parch = data_frame.at[i, 'Parch']
        passenger_record['PassengerId'] = data_frame.at[i, 'PassengerId']

        if sibsp == 0 and parch == 0:
            passenger_record['FamilyType'] = common_const.family_type_ord_encoding['single_no_children']
        elif sibsp > 0 and parch == 0:
            if data_frame.at[i, 'Age'] <= 18:
                passenger_record['FamilyType'] = common_const.family_type_ord_encoding['child']
            else:
                passenger_record['FamilyType'] = common_const.family_type_ord_encoding['couple_no_children']
        elif sibsp > 0 and parch > 0:
            if data_frame.at[i, 'Age'] <= 18:
                passenger_record['FamilyType'] = common_const.family_type_ord_encoding['child']
            else:
                passenger_record['FamilyType'] = common_const.family_type_ord_encoding['couple_with_children']
        else:
            passenger_record['FamilyType'] = common_const.family_type_ord_encoding['single_carer']
        family_df = family_df.append(passenger_record, ignore_index=True)

    return family_df


def adjust_fare_per_person(data_frame):
    df_duplicated_tickets = data_frame.pivot_table(index=['Ticket'], aggfunc='size')

    for i in data_frame.index:
        if df_duplicated_tickets[data_frame.at[i, 'Ticket']] > 1:
            data_frame.at[i, 'Fare'] = round(
                data_frame.at[i, 'Fare']/df_duplicated_tickets[data_frame.at[i, 'Ticket']], 2)


def extract_deck(data_frame):
    decks_df = pandas.DataFrame()
    passenger_record = {
        'PassengerId': 0,
        'Deck': 0
    }

    for i in data_frame.index:
        passenger_record['PassengerId'] = data_frame.at[i, 'PassengerId']

        if pandas.isna(data_frame.at[i, 'Cabin']):
            passenger_record['Deck'] = common_const.deck_ord_encoding['U']
        else:
            passenger_record['Deck'] = common_const.deck_ord_encoding[str(data_frame.at[i, 'Cabin'])[0]]

        decks_df = decks_df.append(passenger_record, ignore_index=True)

    return decks_df


def extract_ticket_code(data_frame):
    for i in data_frame.index:
        if str(data_frame.at[i, 'Ticket']).lower().islower():  # check if contains any non-numbers
            ticket_id = sub("[^a-zA-Z]", "", data_frame.at[i, 'Ticket'])[0:1]
            data_frame.at[i, 'Ticket'] = ticket_id
        else:
            ticket_id = "NA"
            data_frame.at[i, 'Ticket'] = ticket_id

    dummies = pandas.get_dummies(data_frame['Ticket'])
