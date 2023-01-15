import pandas as pd
import requests
from parse_html_portal_da import parse_html, parse_details
from tqdm import tqdm
import time


path_to_links = 'portal_da_links_lands_10_01_23_2.xlsx'
path_to_unique_keys = 'unique_cdata_keys.xlsx'
path_to_needed_keys = 'needed_cdata_keys.xlsx'

agent_str = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'


links_df = pd.read_excel(path_to_links)
unique_keys_df = pd.read_excel('unique_cdata_keys.xlsx')
needed_keys_df = pd.read_excel('needed_cdata_keys.xlsx')

new_keys_to_append = pd.DataFrame(columns=['new_keys', 'links'])  # to save new keys from htmls
result_df = pd.DataFrame(columns=['links'] + needed_keys_df['keys'].to_list() + ['views', 'region_str',
                                                                                 'publication_date'])

try:
    for link in tqdm(links_df['links']):
        try:
            r = requests.get(link, headers={'user-agent': agent_str})
            html_text = r.text
            result_dict, new_unique_keys = parse_html(html_text, unique_keys_df, needed_keys_df)
            views, region_str, publication_date = parse_details(html_text)

            if len(new_unique_keys) > 0:
                new_keys_to_append = pd.concat([pd.DataFrame([{'new_keys': new_unique_keys, 'links': link}]),
                                                new_keys_to_append], ignore_index=True)

            df_to_append = pd.DataFrame([result_dict])
            df_to_append['links'], df_to_append['views'], df_to_append['region_str'], df_to_append['publication_date'] \
            = link, views, region_str, publication_date

            result_df = pd.concat([result_df, df_to_append], ignore_index=True)
            time.sleep(1)
        except Exception as e:
            print(f'Exception on loop {e}\nsleep 5 secs...')
            time.sleep(5)
except Exception as e:
    print(f'General Exception {e}')

result_df.to_excel(r'data/links_results_15_01_23.xlsx', index=False)

if len(new_keys_to_append) > 0:
    print(f'New keys in {len(new_keys_to_append)} links')
    new_keys_to_append.to_excel(r'data/new_keys_15_01_23.xlsx', index=False)














