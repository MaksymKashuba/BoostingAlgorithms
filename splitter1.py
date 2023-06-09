import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('C:/Users/max19/PycharmProjects/ML_Ex/data/adult.data')

data['income'] = data['income'].replace([' <=50K', ' >50K'], [0, 1])
data['education'] = data['education'].replace([' Bachelors', ' Masters', ' 9th', ' Some-college', ' Assoc-acdm',
                                               ' Assoc-voc',
                                               ' 7th-8th', ' Doctorate', ' Prof-school', ' 5th-6th', ' 10th',
                                               ' 1st-4th',
                                               ' Preschool', ' 12th', ' 11th', ' HS-grad'],
                                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

data['marital-status'] = data['marital-status'].replace(['Never-married', 'Married-civ-spouse', 'Divorced',
                                               'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed'],
                                              [0, 1, 2, 3, 4, 5, 6])

data['occupation'] = data['occupation'].replace([' Adm-clerical', ' Exec-managerial', ' Handlers-cleaners',
                                                         ' Prof-specialty', ' Other-service', ' Sales', ' Craft-repair',
                                                         ' Transport-moving', ' Farming-fishing', ' Machine-op-inspct',
                                                         ' Tech-support', ' ?', ' Protective-serv', ' Armed-Forces',
                                                         ' Priv-house-serv'],
                                                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
X = data.drop('income', axis=1)
Y = data['income']

X_encoded = pd.get_dummies(X)
X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=1)
print(list(X_train.columns))