import pandas as pd
import requests
from parse_html_portal_da import parse_html
from tqdm import tqdm
import time


path_to_links = 'portal_da_links_lands_10_01_23_2.xlsx'
path_to_unique_keys = 'unique_cdata_keys.xlsx'
path_to_needed_keys = 'needed_cdata_keys.xlsx'

agent_str = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'


links_df = pd.read_excel(path_to_links)
unique_keys_df = pd.read_excel('unique_cdata_keys.xlsx')
needed_keys_df = pd.read_excel('needed_cdata_keys.xlsx')

new_keys_to_append = []  # to save new keys from htmls
result_df = pd.DataFrame(columns=['links'] + needed_keys_df['keys'].to_list())  # dataframe to save data

try:
    for link in tqdm(links_df[:10]['links']):
        try:
            r = requests.get(link, headers={'user-agent': agent_str})
            html_text = r.text
            result_dict, new_unique_keys = parse_html(html_text, unique_keys_df, needed_keys_df)

            if len(new_unique_keys) > 0:
                new_keys_to_append.append(new_unique_keys)

            df_to_append = pd.DataFrame([result_dict])
            df_to_append['links'] = link
            result_df = pd.concat([result_df, df_to_append], ignore_index=True)
            time.sleep(1)
        except Exception as e:
            print(f'Exception on loop {e}\nsleep 5 secs...')
            time.sleep(5)
except Exception as e:
    print(f'General Exception {e}')

result_df.to_excel('links_results_12_01_23.xlsx', index=False)

if len(set(new_keys_to_append)) > 0:
    print(f'There are {len(set(new_keys_to_append))} new keys!')
    new_keys_df = pd.DataFrame()
    new_keys_df['new_keys'] = list(set(new_keys_to_append))
    new_keys_df.to_excel('new_keys_12_01_23.xlsx')















