import pandas as pd
import numpy as np
from datetime import datetime as dt
import ast
from rosreestr2coord import Area
from tqdm.auto import tqdm
from geopy.geocoders import Nominatim
import re

tqdm.pandas()
geolocator = Nominatim(user_agent="my app")
#pd.options.display.float_format = '{:.2f}'.format

IN_DATA_PATH = r'../data/lien_data_Jan.csv'
OUT_DATA_FOLDER = r'data/'


def master_data(in_data_df):
    """
    Function to explode data from jsons in parsing results from lien portal
    :param in_data_df: dataframe with data
    :return:
    """
    print(f'Start master_data function {dt.now().strftime("%H:%M:%S")}')
    links_df = in_data_df.copy()
    links_df.columns = [x.replace('general.', '') for x in links_df.columns]
    links_df['asset_id'] = links_df['asset_id'].apply(lambda x: str(x).replace('"', ''))

    # explode data from jsons
    print(f'explode data from jsons {dt.now().strftime("%H:%M:%S")}')
    links_df['contact_name'] = links_df['contact'].apply(lambda x: ast.literal_eval(x)['name']
                                                                    if pd.notnull(x) else np.NaN)
    links_df['contact_email'] = links_df['contact'].apply(lambda x: ast.literal_eval(x)['email']
                                                                    if pd.notnull(x) else np.NaN)
    links_df['seller_name'] = links_df['seller'].apply(lambda x: ast.literal_eval(x)['name']
                                                                    if pd.notnull(x) else np.NaN)

    # handle with specific fields
    print(f'handle with specific fields {dt.now().strftime("%H:%M:%S")}')
    applied_df = links_df.loc[links_df['specific_fields'].str.len() > 3].\
                 apply(lambda row: handle_specific_fields(row.specific_fields), axis='columns', result_type='expand')
    links_df = pd.concat([links_df, applied_df], axis=1)

    # handle prices with regex
    print(f'handle prices with regex {dt.now().strftime("%H:%M:%S")}')
    applied_df = pd.DataFrame()
    applied_df['price_value'] = links_df.loc[links_df['price'].str.len() > 3, 'price'].apply(lambda x: extract_price(x))
    links_df = pd.concat([links_df, applied_df], axis=1)

    # append geo information from cadastral number
    print(f'append geo information from cadastral number {dt.now().strftime("%H:%M:%S")}')
    applied_df = pd.DataFrame()
    applied_df['cadastral_geocode'] = links_df.loc[links_df['cadastral_number'].str.len() > 3,
                                                   'cadastral_number'].progress_apply(lambda x: get_kadastr_data(x))
    links_df = pd.concat([links_df, applied_df], axis=1)

    # append geo information from string name of region
    print(f'append geo information from string name of region {dt.now().strftime("%H:%M:%S")}')
    applied_df = pd.DataFrame()
    applied_df['region_geocode'] = links_df.loc[links_df['region_str'].str.len() > 3, 'region_str'].progress_apply(
        lambda x: parse_out_region_str(x))
    links_df = pd.concat([links_df, applied_df], axis=1)

    # explode square from area
    print(f'explode square from area {dt.now().strftime("%H:%M:%S")}')
    links_df['square'] = links_df['area'].apply(lambda x: explode_area(x))
    links_df['views'] = links_df['views'].apply(lambda x: re.sub('\D+', '', str(x)))
    links_df['views'] = links_df['views'].apply(pd.to_numeric)

    # filtered data
    refined_df = links_df[['asset_id', 'views', 'publication_date', 'contact_name', 'seller_name', 'square',
                           'price_value', 'cadastral_geocode', 'region_geocode']]
    print(f'finish')
    return links_df, refined_df


def explode_area(x):
    """
    Function to explode out square from are string
    :param x: row
    :return:
    """
    return float(str(x).replace('сот.', '').replace(' ', ''))*100


def handle_specific_fields(x):
    """
    Function to apply to specific_fields section
    :param x: row
    :return: new dataframe with values from specific fields
    """
    buf_el = ast.literal_eval(x)
    return_values = {}

    for el in buf_el:
        return_values[el['field']] = el['value']
    return return_values


def extract_price(x):
    """
    Simple function to extract price data with regex
    :param x: row
    :return:
    """
    x = str(x).lower()
    if 'млн' in x:
        return float(x.lower().replace('млн', '').replace('₽', '').replace(' ', '').replace('"', '').\
                                                                                    replace(',', '.'))*1_000_000
    elif 'млрд' in x:
        return float(x.lower().replace('млрд', '').replace('₽', '').replace(' ', '').replace('"', '').\
                                                                                     replace(',', '.'))*1_000_000_000
    else:
        return float(x.replace(' ', '').replace('₽', '').replace('"', '').replace(',', ''))


def get_kadastr_data(kadastr_str):
    """
    Function to load kadastr info
    :param kadastr_str: string with kadastr code
    :return:
    """
    try:
        area = Area(kadastr_str, with_log=False)
        area_json = area.get_center_xy()[0][0][0]
        area_json.reverse()
    except Exception as e:
        area_json = str(e)
    return area_json


def parse_out_region_str(region_str):
    """
    Function to get geocode of region_str
    :param region_str:
    :return:
    """
    try:
        raw_info = geolocator.geocode(region_str).raw
    except:
        return np.NaN
    return [float(raw_info['lat']), float(raw_info['lon'])]


in_data_df = pd.read_csv(IN_DATA_PATH, sep=';')
wide_df, refined_df = master_data(in_data_df)

wide_df.to_csv(fr'{OUT_DATA_FOLDER}/wide_df_{dt.now().strftime("%H_%M_%d_%m_%Y")}.csv', index=False, sep=';')
refined_df.to_csv(fr'{OUT_DATA_FOLDER}/refined_df_{dt.now().strftime("%H_%M_%d_%m_%Y")}.csv', index=False, sep=';')







