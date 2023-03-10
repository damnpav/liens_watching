import pandas as pd
import re
import requests


# unique_keys_df = pd.read_excel('unique_cdata_keys.xlsx')
# needed_keys_df = pd.read_excel('needed_cdata_keys.xlsx')


def parse_html(html_text, unique_keys_df, needed_keys_df):
    """
    Function to parse cdata section from html files from portal da
    :param html_text: html str
    :param unique_keys_df: dataframe with seen unique keys
    :param needed_keys_df: dataframe with keys to filter out
    :return: result_dict (dict); new_unique_keys (list) - list with new unseen keys
    """
    cdata_str = html_text[html_text.find('CDATA') + 7:html_text.find('//]]>') - 1]
    cdata_list = cdata_str.split(';')
    cdata_keys = [x.split('=')[0] for x in cdata_list if x.split('=')[0] != '']
    new_unique_keys = list(set(cdata_keys) - set(unique_keys_df['keys']))
    new_unique_keys = [x.replace('{', '').replace('}', '') for x in new_unique_keys]

    # filter out selected data
    selected_data = [x for x in cdata_list if x.split('=')[0] in needed_keys_df['keys'].to_list()]
    data_keys = [x.split('=')[0] for x in selected_data]
    data_values = [x.split('=')[1] for x in selected_data]

    result_dict = dict(zip(data_keys, data_values))
    return result_dict, new_unique_keys


def save_html(html_str, path_to_save):
    agent_str = f'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/' \
                f'91.0.4472.114 Safari/537.36'
    r = requests.get(html_str, headers={'user-agent': agent_str})
    html_text = r.text
    with open(path_to_save, 'w') as html_file:
        html_file.write(html_text)
    return html_text


def parse_details(html_text):
    """
    Function to parse out details of lien from html code
    :param html_text: goal html text
    :return:
    """
    error_list = []
    views, region_str, publication_date = '', '', ''

    try:
        views = re.findall('\d+.просмотр', html_text)[0]
    except:
        error_list.append('views')

    try:
        region_str = re.findall('mr-4.*</span><span', html_text)[0].replace('mr-4">', '').replace('</span><span', '')
    except:
        error_list.append('region_str')

    try:
        publication_date = re.findall('time-s.*\d\d\s\S{3,8}\s\d\d\d\d', html_text)[0].replace('time-s"></ui-icon>',
                                                                                                '')
    except:
        error_list.append('publication_date')

    if len(error_list) > 0:
        print(f'Errors in {error_list}')

    return views, region_str, publication_date






