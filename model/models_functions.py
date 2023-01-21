import pandas as pd
from sklearn.preprocessing import LabelEncoder


data_path = '../lien_df1_18_01_23.csv'


def encode_features(selected_df, features_list):
    """
    Function to encode categorical data
    :param features_list: list with categorical features to encode
    :param selected_df: input data
    :return:
    """
    le = LabelEncoder()
    label_encoded_df = selected_df[features_list].copy()
    for col in label_encoded_df.select_dtypes(include='O').columns:
        label_encoded_df[col]=le.fit_transform(label_encoded_df[col])
    return label_encoded_df












