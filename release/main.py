import pandas as pd
from models_functions import model_conveyor


lien_df = pd.read_csv('../data/lien_df1_29_01_23_filled_na_zeros.csv', sep=';')
lien_df = lien_df.loc[lien_df['price_value'] < lien_df['price_value'].quantile(.87)]  # обрезаем персентиль


# фильтруем фичи
selected_df = lien_df[['views', 'contact_name', 'seller_name', 'square', 'cadastral_geocode',
                       'region_geocode','shown_by_days', 'distance', 'region_square', 'population',
                       'region_capital', 'grp', 'criminality', 'km_to_msk', 'region_capital_geocode',
                       'km_to_region_capital',  'price_value']].copy()

selected_df = selected_df.dropna()  # нуллы предварительно уже обработали

# используемые метрики
metrics_names = ['mape', 'mae', 'explained_variance_score', 'max_error', 'mse',
                 'r2', 'd2_ae', 'd2_pinball', 'd2_tweedie']

# используемые шакалеры
scalers_names = ['minmax', 'stdsc']



# проверяем результаты для разных скейлеров на всех метриках для линейной регрессии и регрессии с полиномиальными элементами
# опытным путем лучшие результаты выявлены на степени degree =2
for scaler_name in scalers_names:
    print(scaler_name)
    print('Making Linear Regression')
    model, metrics_df, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, \
    scaler_y_train, scaler_x_test, scaler_y_test = \
        model_conveyor(selected_df, 'linear_regression', scaler_name, metrics_names, True, 'model_dumps/')

    print('Making Linear Regression with Polynomial features')
    model, metrics_df, X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, scaler_x_train, \
    scaler_y_train, scaler_x_test, scaler_y_test = \
        model_conveyor(selected_df, 'linear_regression_with_poly_features', scaler_name,
                       metrics_names, True, 'model_dumps/', poly_degree=2)




