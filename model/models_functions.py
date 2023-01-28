import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data_path = '../data/lien_df1_18_01_23.csv'


def encode_features(selected_df):
    """
    Function to encode categorical data
    :param selected_df: input data
    :return:
    """
    le = LabelEncoder()
    label_encoded_df = selected_df.copy()
    for col in label_encoded_df.select_dtypes(include='O').columns:
        label_encoded_df[col] = le.fit_transform(label_encoded_df[col])
    return label_encoded_df


def make_regression(data_df, target_col):
    """
    Function to fit regression model to data
    :param data_df: dataframe with learning data
    :param target_col: column with target
    :return: model (object of regr model), X_test_std, Y_test_std (scaled test data), stdsc (object of Scaler)
    """

    # splitting data in train and test datasets, also in target and features (X,Y)
    train_df, test_df = train_test_split(data_df, test_size=0.2, shuffle=True)
    train_X = train_df[list(set(train_df.columns) - set([target_col]))]
    train_Y = train_df[target_col]
    test_X = test_df[list(set(test_df.columns) - set([target_col]))]
    test_Y = test_df[target_col]

    # objects of model and scaler
    regression_model = LinearRegression()
    stdsc = StandardScaler()
    # todo
    # ты fit transofrm для разных данных используешь, так нельзя!

    # scale data
    X_train_std = stdsc.fit_transform(train_X)
    Y_train_std = stdsc.fit_transform(np.array(train_Y).reshape(-1, 1))
    X_test_std = stdsc.fit_transform(test_X)
    Y_test_std = stdsc.fit_transform(np.array(test_Y).reshape(-1, 1))


    # fit model
    model = regression_model.fit(X_train_std, Y_train_std)

    # todo c этим надо еще поработать - навесить функцию тестирования, показа результатов и тд

    return model, X_test_std, Y_test_std, stdsc













