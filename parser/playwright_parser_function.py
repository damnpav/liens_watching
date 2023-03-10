from playwright.sync_api import sync_playwright
from datetime import datetime as dt
import pandas as pd
import time

links = []
#section_link = 'https://portal-da.ru/objects/catalog/buy/warehouse_complex'
section_link = 'https://portal-da.ru/objects/catalog/buy/section'


def click_and_safe(section_link, links):
    """
    Function to click and save links from sections of lien portal
    :param section_link: link to section
    :param links: list where to save links
    :return:
    """
    try:
        with sync_playwright() as p:
            browser_type = p.chromium
            browser = browser_type.launch()
            page = browser.new_page()
            print('Open main tab')
            page.goto(section_link)
            time.sleep(3)


            # while button exist - click it
            flag = True
            k = 0
            while flag:
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
                        print(f'Clicking anoter offer: {k}')
                        with page.expect_popup() as page1_info:
                            page.locator(
                                f"div:nth-child({k}) > .asset-card__content-wrapper > .asset-card__photos > a").click()
                        page1 = page1_info.value
                        current_link = page1.url
                        links.append(current_link)
                    k += 1
                except Exception as e:
                    print('All links are clicked')
                    print(f'Exception: {e}')
                    flag = False
    except Exception as e:
        print(f'Global exception: {e}')
    return links

print(f'start')
result_df = pd.DataFrame()
result_df['links'] = click_and_safe(section_link, links)
result_df.to_excel('portal_da_links_lands_10_01_23_2.xlsx', index=False)

# TODO - done, now need in parser per link


