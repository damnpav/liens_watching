import pandas as pd
import plotly.express as px
import numpy as np
from tqdm import tqdm
from datetime import datetime as dt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, PowerTransformer, PolynomialFeatures
import joblib


pd.options.plotting.backend = "plotly"

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


def make_regression(data_df, target_col, scaler_name):
    """
    Function to fit regression model to data
    :param scaler_name: name of scaler to scale data
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

    # objects of model
    regression_model = LinearRegression(positive=True)

    # scalers
    scaler_x_train, scaler_y_train, scaler_x_test, scaler_y_test = get_scalers(scaler_name)

    # scale data
    X_train_scaled = scaler_x_train.fit_transform(train_X)
    Y_train_scaled = scaler_y_train.fit_transform(np.array(train_Y).reshape(-1, 1))
    X_test_scaled = scaler_x_test.fit_transform(test_X)
    Y_test_scaled = scaler_y_test.fit_transform(np.array(test_Y).reshape(-1, 1))

    # fit model
    model = regression_model.fit(X_train_scaled, Y_train_scaled)

    return model, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, scaler_y_train, \
           scaler_x_test, scaler_y_test


def make_polynomial_regression(data_df, target_col, scaler_name, poly_degree):
    # todo надо проработать
    """
    Function to make polynomial regression
    :param data_df:
    :param target_col:
    :param scaler_name:
    :return:
    """
    # splitting data in train and test datasets, also in target and features (X,Y)
    train_df, test_df = train_test_split(data_df, test_size=0.2, shuffle=True)
    train_X = train_df[list(set(train_df.columns) - set([target_col]))]
    train_Y = train_df[target_col]
    test_X = test_df[list(set(test_df.columns) - set([target_col]))]
    test_Y = test_df[target_col]

    # objects of model
    regression_model = LinearRegression(positive=True)

    # scalers
    scaler_x_train, scaler_y_train, scaler_x_test, scaler_y_test = get_scalers(scaler_name)

    # scale data
    X_train_scaled = scaler_x_train.fit_transform(train_X)
    Y_train_scaled = scaler_y_train.fit_transform(np.array(train_Y).reshape(-1, 1))
    X_test_scaled = scaler_x_test.fit_transform(test_X)
    Y_test_scaled = scaler_y_test.fit_transform(np.array(test_Y).reshape(-1, 1))

    polynomial = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_train_scaled = polynomial.fit_transform(X_train_scaled)
    X_test_scaled = polynomial.fit_transform(X_test_scaled)


    # fit model
    model = regression_model.fit(X_train_scaled, Y_train_scaled)

    return model, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, scaler_y_train, \
           scaler_x_test, scaler_y_test


def get_scalers(your_scaler):
    """
    Function to get different scalers
    :param your_scaler: name of your choosen scaler
    :return: scalers for x_train, y_train, x_test, y_test
    """
    if your_scaler == 'minmax':
        scaler_x_train = MinMaxScaler()
        scaler_y_train = MinMaxScaler()
        scaler_x_test = MinMaxScaler()
        scaler_y_test = MinMaxScaler()
    elif your_scaler == 'stdsc':
        scaler_x_train = StandardScaler()
        scaler_y_train = StandardScaler()
        scaler_x_test = StandardScaler()
        scaler_y_test = StandardScaler()
    elif your_scaler == 'norm':
        scaler_x_train = Normalizer()
        scaler_y_train = Normalizer()
        scaler_x_test = Normalizer()
        scaler_y_test = Normalizer()
    elif your_scaler == 'power':
        scaler_x_train = PowerTransformer()
        scaler_y_train = PowerTransformer()
        scaler_x_test = PowerTransformer()
        scaler_y_test = PowerTransformer()
    else:
        pass
    return scaler_x_train, scaler_y_train, scaler_x_test, scaler_y_test


def get_metrics(model, scaler_y_train, scaler_y_test, X_train_scaled, Y_train_scaled, X_test_scaled,
                Y_test_scaled, metric_name):
    """
    Function to metric predictions of model
    There are 14 metrics: mape, mae, explained_variance_score, max_error, mse, rmse, msle, rmsle, r2, tweedie,
    gamma, d2_ae, d2_pinball, metric
    :param model: object of model
    :param scaler_y_train: scalers
    :param scaler_y_test:
    :param X_train_scaled: data
    :param Y_train_scaled:
    :param X_test_scaled:
    :param Y_test_scaled:
    :param metric_name:
    :return:
    """

    if metric_name == 'mape':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        metric_train_scaled = metrics.mean_absolute_percentage_error(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        metric_test_scaled = metrics.mean_absolute_percentage_error(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        metric_train_unscaled = metrics.mean_absolute_percentage_error(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        metric_test_unscaled = metrics.mean_absolute_percentage_error(Y_test_unscaled, y_test_unscaled_pred)
        return metric_train_scaled, metric_test_scaled, metric_train_unscaled, metric_test_unscaled
    elif metric_name == 'mae':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        mae_train_scaled = metrics.mean_absolute_error(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        mae_test_scaled = metrics.mean_absolute_error(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        mae_train_unscaled = metrics.mean_absolute_error(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        mae_test_unscaled = metrics.mean_absolute_error(Y_test_unscaled, y_test_unscaled_pred)
        return mae_train_scaled, mae_test_scaled, mae_train_unscaled, mae_test_unscaled
    elif metric_name == 'explained_variance_score':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        evs_train_scaled = metrics.explained_variance_score(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        evs_test_scaled = metrics.explained_variance_score(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        evs_train_unscaled = metrics.explained_variance_score(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        evs_test_unscaled = metrics.explained_variance_score(Y_test_unscaled, y_test_unscaled_pred)
        return evs_train_scaled, evs_test_scaled, evs_train_unscaled, evs_test_unscaled
    elif metric_name == 'max_error':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        max_error_train_scaled = metrics.max_error(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        max_error_test_scaled = metrics.max_error(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        max_error_train_unscaled = metrics.max_error(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        max_error_test_unscaled = metrics.max_error(Y_test_unscaled, y_test_unscaled_pred)
        return max_error_train_scaled, max_error_test_scaled, max_error_train_unscaled, max_error_test_unscaled
    elif metric_name == 'mse':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        mse_train_scaled = metrics.mean_squared_error(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        mse_test_scaled = metrics.mean_squared_error(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        mse_train_unscaled = metrics.mean_squared_error(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        mse_test_unscaled = metrics.mean_squared_error(Y_test_unscaled, y_test_unscaled_pred)
        return mse_train_scaled, mse_test_scaled, mse_train_unscaled, mse_test_unscaled
    elif metric_name == 'rmse':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        mse_train_scaled = metrics.mean_squared_error(Y_train_scaled, y_train_scaled_pred, squared=False)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        mse_test_scaled = metrics.mean_squared_error(Y_test_scaled, y_test_scaled_pred, squared=False)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        mse_train_unscaled = metrics.mean_squared_error(Y_train_unscaled, y_train_unscaled_pred, squared=False)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        mse_test_unscaled = metrics.mean_squared_error(Y_test_unscaled, y_test_unscaled_pred, squared=False)
        return mse_train_scaled, mse_test_scaled, mse_train_unscaled, mse_test_unscaled
    elif metric_name == 'msle':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        msle_train_scaled = metrics.mean_squared_log_error(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        msle_test_scaled = metrics.mean_squared_log_error(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        msle_train_unscaled = metrics.mean_squared_log_error(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        msle_test_unscaled = metrics.mean_squared_log_error(Y_test_unscaled, y_test_unscaled_pred)
        return msle_train_scaled, msle_test_scaled, msle_train_unscaled, msle_test_unscaled
    elif metric_name == 'rmsle':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        rmsle_train_scaled = metrics.mean_squared_log_error(Y_train_scaled, y_train_scaled_pred, squared=False)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        rmsle_test_scaled = metrics.mean_squared_log_error(Y_test_scaled, y_test_scaled_pred, squared=False)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        rmsle_train_unscaled = metrics.mean_squared_log_error(Y_train_unscaled, y_train_unscaled_pred, squared=False)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        rmsle_test_unscaled = metrics.mean_squared_log_error(Y_test_unscaled, y_test_unscaled_pred, squared=False)
        return rmsle_train_scaled, rmsle_test_scaled, rmsle_train_unscaled, rmsle_test_unscaled
    elif metric_name == 'r2':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        r2_train_scaled = metrics.r2_score(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        r2_test_scaled = metrics.r2_score(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        r2_train_unscaled = metrics.r2_score(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        r2_test_unscaled = metrics.r2_score(Y_test_unscaled, y_test_unscaled_pred)
        return r2_train_scaled, r2_test_scaled, r2_train_unscaled, r2_test_unscaled
    elif metric_name == 'tweedie':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        tweedie_train_scaled = metrics.mean_poisson_deviance(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        tweedie_test_scaled = metrics.mean_poisson_deviance(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        tweedie_train_unscaled = metrics.mean_poisson_deviance(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        tweedie_test_unscaled = metrics.mean_poisson_deviance(Y_test_unscaled, y_test_unscaled_pred)
        return tweedie_train_scaled, tweedie_test_scaled, tweedie_train_unscaled, tweedie_test_unscaled
    elif metric_name == 'gamma':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        gamma_train_scaled = metrics.mean_gamma_deviance(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        gamma_test_scaled = metrics.mean_gamma_deviance(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        gamma_train_unscaled = metrics.mean_gamma_deviance(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        gamma_test_unscaled = metrics.mean_gamma_deviance(Y_test_unscaled, y_test_unscaled_pred)
        return gamma_train_scaled, gamma_test_scaled, gamma_train_unscaled, gamma_test_unscaled
    elif metric_name == 'd2_ae':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        d2_ae_train_scaled = metrics.d2_absolute_error_score(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        d2_ae_test_scaled = metrics.d2_absolute_error_score(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        d2_ae_train_unscaled = metrics.d2_absolute_error_score(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        d2_ae_test_unscaled = metrics.d2_absolute_error_score(Y_test_unscaled, y_test_unscaled_pred)
        return d2_ae_train_scaled, d2_ae_test_scaled, d2_ae_train_unscaled, d2_ae_test_unscaled
    elif metric_name == 'd2_pinball':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        d2_pinball_train_scaled = metrics.d2_pinball_score(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        d2_pinball_test_scaled = metrics.d2_pinball_score(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        d2_pinball_train_unscaled = metrics.d2_pinball_score(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        d2_pinball_test_unscaled = metrics.d2_pinball_score(Y_test_unscaled, y_test_unscaled_pred)
        return d2_pinball_train_scaled, d2_pinball_test_scaled, d2_pinball_train_unscaled, d2_pinball_test_unscaled
    elif metric_name == 'd2_tweedie':
        # scaled data
        # train data
        y_train_scaled_pred = model.predict(X_train_scaled)
        d2_tweedie_train_scaled = metrics.d2_tweedie_score(Y_train_scaled, y_train_scaled_pred)

        # test data
        # train data
        y_test_scaled_pred = model.predict(X_test_scaled)
        d2_tweedie_test_scaled = metrics.d2_tweedie_score(Y_test_scaled, y_test_scaled_pred)

        # unscaled data
        # train data
        y_train_unscaled_pred = scaler_y_train.inverse_transform(y_train_scaled_pred)
        Y_train_unscaled = scaler_y_train.inverse_transform(Y_train_scaled)
        d2_tweedie_train_unscaled = metrics.d2_tweedie_score(Y_train_unscaled, y_train_unscaled_pred)

        # test data
        y_test_unscaled_pred = scaler_y_test.inverse_transform(y_test_scaled_pred)
        Y_test_unscaled = scaler_y_train.inverse_transform(Y_test_scaled)
        d2_tweedie_test_unscaled = metrics.d2_tweedie_score(Y_test_unscaled, y_test_unscaled_pred)
        return d2_tweedie_train_scaled, d2_tweedie_test_scaled, d2_tweedie_train_unscaled, d2_tweedie_test_unscaled
    else:
        print("""There's no such name of metric name\nPossible metrics: metric, mape, explained_variance_score, 
                 max_error, mse, rmse, msle, rmsle, r2, tweedie, gamma, d2_ae, d2_pinball, metric""")
        return None


def plot_test_results(model, X_test_scaled, Y_test_scaled, scaler_y_test, path_to_save):
    """
    Model to plot test values
    :param path_to_save: path where to save html file
    :param model: object of model
    :param X_test_scaled: test features
    :param Y_test_scaled: test target
    :param scaler_y_test: scaler
    :return:
    """
    pred_df = pd.DataFrame(scaler_y_test.inverse_transform(model.predict(X_test_scaled)))
    true_df = pd.DataFrame(scaler_y_test.inverse_transform(Y_test_scaled))
    pred_df = pred_df.rename(columns={0: 'prediction_value'})
    true_df = true_df.rename(columns={0: 'true_value'})
    joined_df = pred_df.join(true_df)
    joined_df = joined_df.sort_values(by='true_value')
    joined_df = joined_df.reset_index(drop=True)
    fig = joined_df.plot(x=joined_df.index, y=['prediction_value', 'true_value'], title='Test data')
    fig.write_html(path_to_save)


def plot_train_results(model, X_train_scaled, Y_train_scaled, scaler_y_train, path_to_save):
    """
    Model to plot test values
    :param path_to_save: path where to save html file
    :param model: object of model
    :param X_train_scaled: train features
    :param Y_train_scaled: train target
    :param scaler_y_test: scaler
    :return:
    """
    pred_df = pd.DataFrame(scaler_y_train.inverse_transform(model.predict(X_train_scaled)))
    true_df = pd.DataFrame(scaler_y_train.inverse_transform(Y_train_scaled))
    pred_df = pred_df.rename(columns={0: 'prediction_value'})
    true_df = true_df.rename(columns={0: 'true_value'})
    joined_df = pred_df.join(true_df)
    joined_df = joined_df.sort_values(by='true_value')
    joined_df = joined_df.reset_index(drop=True)
    fig = joined_df.plot(x=joined_df.index, y=['prediction_value', 'true_value'], title='Train data')
    fig.write_html(path_to_save)



def model_conveyor(data_df, model_name, scaler_name, metric_names, save_model=False, folder_to_save=None, **kwargs):
    """
    Function to pass data through conveyor of data scaling, encoding and then fitting model and measure test metrics
    :param folder_to_save: where to save model
    :param save_model: If True then function will save models in pkl file
    :param metric_names: list with metrics
    :param scaler_name: name of scaler
    :param data_df: DataFrame with data to model
    :return:
    """
    encoded_df = encode_features(data_df)

    if model_name == 'linear_regression':
        model, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, scaler_y_train, \
        scaler_x_test, scaler_y_test = make_regression(encoded_df, 'price_value', scaler_name)

    elif model_name == 'linear_regression_with_poly_features':
        model, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, scaler_y_train, \
        scaler_x_test, scaler_y_test = make_polynomial_regression(encoded_df, 'price_value', scaler_name,
                                                                  kwargs['poly_degree'])
    else:
        print('model name not found')
        return None

    metrics_results = {}
    for metric_name in tqdm(metric_names):
        print(metric_name)
        metrics_results[metric_name] = {}  # where to store results

        metric_train_scaled, metric_test_scaled, metric_train_unscaled, metric_test_unscaled = \
            get_metrics(model, scaler_y_train, scaler_y_test, X_train_scaled, Y_train_scaled, X_test_scaled,
                        Y_test_scaled, metric_name)

        metrics_results[metric_name][f'train_scaled'] = metric_train_scaled
        metrics_results[metric_name][f'test_scaled'] = metric_test_scaled
        metrics_results[metric_name][f'train_unscaled'] = metric_train_unscaled
        metrics_results[metric_name][f'test_unscaled'] = metric_test_unscaled
        metrics_df = pd.DataFrame(metrics_results)

    # save model
    if save_model:
        joblib.dump(model, fr'{folder_to_save}{model_name}_{scaler_name}_{dt.now().strftime("%H_%M_%d_%m_%Y")}.pkl')
        metrics_df.to_excel(
            fr'{folder_to_save}metrics_{model_name}_{scaler_name}_{dt.now().strftime("%H_%M_%d_%m_%Y")}.xlsx',
            index=True)
        plot_test_results(model, X_test_scaled, Y_test_scaled, scaler_y_test,
                          fr'{folder_to_save}{model_name}_TEST_{scaler_name}_{dt.now().strftime("%H_%M_%d_%m_%Y")}.html')
        plot_train_results(model, X_train_scaled, Y_train_scaled, scaler_y_train,
                           fr'{folder_to_save}{model_name}_TRAIN_{scaler_name}_{dt.now().strftime("%H_%M_%d_%m_%Y")}.html')

    return model, metrics_df, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, \
           scaler_y_train, scaler_x_test, scaler_y_test

















