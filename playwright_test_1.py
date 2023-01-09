# from playwright.sync_api import sync_playwright
import time
#
#
# with sync_playwright() as p:
#     browser_type = p.chromium
#     browser = browser_type.launch()
#     page = browser.new_page()
#     page.goto('https://portal-da.ru/objects/catalog/buy/section')
#     time.sleep(2)
#     my_text = page.locator('asset-id__id').text_content()
#     print(my_text)
#     browser.close()
#
# # TODO надо найти необходимые хендлеры

from playwright.sync_api import Page, expect


def test_example(page: Page) -> None:
    page.goto("https://portal-da.ru/objects/catalog/buy/section")
    time.sleep(3)
    with page.expect_popup() as page1_info:
        page.locator("div:nth-child(8) > .asset-card__content-wrapper > .asset-card__photos > a").click()
    page1 = page1_info.value
    return page1.url


my_url = test_example
print(my_url)
