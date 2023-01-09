from playwright.sync_api import sync_playwright
import time

print('start')
k = 0
with sync_playwright() as p:
    print(k)
    for browser_type in [p.chromium, p.firefox, p.webkit]:
        browser = browser_type.launch()
        page = browser.new_page()
        page.goto('https://portal-da.ru/objects/catalog/buy/section')
        time.sleep(5)
        ua = page.query_selector(".user-agent");
        print(ua.inner_html())
        browser.close()
        k += 1
print('finish')



