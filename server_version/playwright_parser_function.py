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
            logging('Open main tab', db_conn, db_cur)
            page.goto(section_link)
            time.sleep(3)

            # while button exist - click it
            flag = True
            k = 0
            while flag and k < 2:
                logging(f'Show more offers {k}', db_conn, db_cur)
                try:
                    page.get_by_role("button", name="Показать ещё").click()
                except Exception as e:
                    logging('All offers are shown', db_conn, db_cur)
                    logging(f'Exception: {e}', db_conn, db_cur)
                    flag = False
                k += 1
                time.sleep(1)

            logging(f'Open first offer', db_conn, db_cur)
            with page.expect_popup() as page1_info:
                page.locator(".asset-card__photos > a").first.click()
            page1 = page1_info.value
            links.append(page1.url)
            page1.close()

            logging(f'Open another offers', db_conn, db_cur)
            flag = True
            k = 1
            while flag:
                time.sleep(1)
                try:
                    logging(f'Clicking anoter offer: {k}', db_conn, db_cur)
                    with page.expect_popup() as page1_info:
                        page.locator(f"div:nth-child({k}) > .asset-card__content-wrapper > .asset-card__photos > a").click()
                    page1 = page1_info.value
                    current_link = page1.url

                    if 'chrome-error' in current_link:
                        logging(f'chrome error here, taking screenshot', db_conn, db_cur)
                        page1.screenshot(path=rf"screens\screenshot_error_{k}.png")
                        page.screenshot(path=rf'screens\screenshot_error_main_tab_{k}.png')
                        logging(f'lets wait and try again', db_conn, db_cur)
                        page1.close()
                        time.sleep(10)
                        logging(f'Clicking another offer: {k}', db_conn, db_cur)
                        with page.expect_popup() as page1_info:
                            page.locator(
                                f"div:nth-child({k}) > .asset-card__content-wrapper > .asset-card__photos > a").click()
                        page1 = page1_info.value
                        current_link = page1.url

                    page1.close()
                    links.append(current_link)
                    save_link(current_link, db_cur, db_conn)
                    k += 1
                except Exception as e:
                    logging('All links are clicked', db_conn, db_cur)
                    logging(f'Exception: {e}', db_conn, db_cur)
                    flag = False
    except Exception as e:
        logging(f'Global exception: {e}', db_conn, db_cur)
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
        cur.execute(f"INSERT INTO links VALUES ('{link_str}')")
        conn.commit()
        print(f'Link {link_str} appended to db')
    except Exception as e:
        print(f'Exception at database writing (save_link): {str(e)}')
    return None


def logging(log_str, cur, conn):
    """
    Log to db
    :param log_str:
    :return:
    """
    try:
        cur.execute(f"INSERT INTO logging VALUES ('{dt.now().strftime('%Y-%m-%d %H-%M-%S')}', '{log_str}')")
        conn.commit()
    except Exception as e:
        print(f'Exception at database writing (logging): {str(e)}')
    return None


def main():
    print('Connecting to db')
    conn = sqlite3.connect(path_to_db)
    cur = conn.cursor()
    logging('start', cur, conn)
    links_list = click_and_safe(section_link, links, conn, cur)
    logging(f'{len(links_list)} links are saved', cur, conn)
    conn.close()


try:
    main()
except Exception as e:
    print(f'Exception in main: {str(e)}')



