from playwright.sync_api import sync_playwright
from datetime import datetime as dt
import pandas as pd
import time
import sqlite3

links = []
#section_link = 'https://portal-da.ru/objects/catalog/buy/warehouse_complex'
# TODO need to select a short section
# TODO dd greedy saving
section_link = 'https://portal-da.ru/objects/catalog/buy/section'
path_to_db = 'lien_db.db'


def click_and_safe(section_link, links, db_conn, db_cur):
    """
    Function to click and save links from sections of lien portal
    :param db_conn: connection to database
    :param db_cur: cursor to database
    :param section_link: link to section
    :param links: list where to save links
    :return:
    """
    try:
        with sync_playwright() as p:
            browser_type = p.chromium
            browser = browser_type.launch(headless=False)
            page = browser.new_page()
            print('Open main tab')
            page.goto(section_link)
            time.sleep(3)

            # while button exist - click it
            flag = True
            k = 0
            while flag and k < 2:
                print(f'Show more offers {k}')
                try:
                    page.get_by_role("button", name="Показать ещё").click()
                except Exception as e:
                    print('All offers are shown')
                    print(f'Exception: {e}')
                    flag = False
                k += 1
                time.sleep(1)

            print(f'Open first offer')
            with page.expect_popup() as page1_info:
                page.locator(".asset-card__photos > a").first.click()
            page1 = page1_info.value
            links.append(page1.url)
            page1.close()

            print(f'Open another offers')
            flag = True
            k = 1
            while flag:
                print(dt.now().strftime('%H:%M:%S'))
                time.sleep(1)
                try:
                    print(f'Clicking anoter offer: {k}')
                    with page.expect_popup() as page1_info:
                        page.locator(f"div:nth-child({k}) > .asset-card__content-wrapper > .asset-card__photos > a").click()
                    page1 = page1_info.value
                    current_link = page1.url



                    if 'chrome-error' not in current_link:
                        links.append(current_link)
                        page1.close()
                    else:
                        print(f'chrome error here, taking screenshot')
                        page1.screenshot(path=rf"screens\screenshot_error_{k}.png")
                        page.screenshot(path=rf'screens\screenshot_error_main_tab_{k}.png')
                        print(f'lets wait and try again')
                        page1.close()
                        time.sleep(10)
                        print(f'Clicking another offer: {k}')
                        with page.expect_popup() as page1_info:
                            page.locator(
                                f"div:nth-child({k}) > .asset-card__content-wrapper > .asset-card__photos > a").click()
                        page1 = page1_info.value
                        current_link = page1.url
                        links.append(current_link)
                        save_link(current_link, db_cur, db_conn)
                    k += 1
                except Exception as e:
                    print('All links are clicked')
                    print(f'Exception: {e}')
                    flag = False
    except Exception as e:
        print(f'Global exception: {e}')
    return links


def save_link(link_str, cur, conn):
    """
    Function to save link from parser to db
    :param conn: connection to database
    :param link_str: string with link
    :param cur: cursor to database
    :return:
    """
    try:
        cur.execute("INSERT INTO links VALUES ('{link_str}')")
        conn.commit()
        print(f'Link {link_str} appended to db')
    except Exception as e:
        print(f'Exception at database writing: {str(e)}')
    return None


print('Connecting to db')
conn = sqlite3.connect(path_to_db)
cur = conn.cursor()


print(f'start')
links_list = click_and_safe(section_link, links, conn, cur)
print(f'{len(links_list)} links are saved')

# print(f'save links to db')
# conn = sqlite3.connect(path_to_db)
# result_df.to_sql('links', conn, if_exists='append')

conn.close()

#result_df.to_excel('portal_da_links_lands_10_01_23_2.xlsx', index=False)

# TODO - done, now need in parser per link


