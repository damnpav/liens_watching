from playwright.sync_api import Page, expect
from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser_type = p.chromium
    browser = browser_type.launch()
    page = browser.new_page()
    page.goto("https://portal-da.ru/objects/catalog/buy/section")
    time.sleep(3)
    with page.expect_popup() as page1_info:
        page.locator("div:nth-child(8) > .asset-card__content-wrapper > .asset-card__photos > a").click()
    page1 = page1_info.value
    print(page1.url)
## got it!! вот так надо доставать url
# теперь надо научиться выводить все объявления и ходить по ним
# или попробовать сразу на старте 'evaluateAll' получить все локаторы



