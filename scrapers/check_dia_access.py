"""Compare ordinary homepage access in two CI clients; no scraping or retries."""
import argparse
import json
from pathlib import Path

URL = 'https://www.dia.es/'
OUT = Path('dia_access_report')
OUT.mkdir(exist_ok=True)


def check_http():
    import requests
    response = requests.get(URL, timeout=30)
    return {'status_code': response.status_code, 'url': response.url,
            'blocked': response.status_code in (401, 403, 429) or 'access denied' in response.text.lower(),
            'body_excerpt': response.text[:1500]}


def check_browser():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    options = Options()
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Standard headed Chrome on the virtual display. No identity overrides.
    driver = webdriver.Chrome(options=options)
    try:
        driver.set_page_load_timeout(45)
        driver.get(URL)
        body = driver.find_element(By.TAG_NAME, 'body').text
        driver.save_screenshot(str(OUT / 'browser.png'))
        text = (driver.title + ' ' + body).lower()
        return {'url': driver.current_url, 'title': driver.title,
                'blocked': any(s in text for s in ('access denied', 'captcha', 'verify you are human', 'confirma que no eres un robot')),
                'body_excerpt': body[:1500]}
    finally:
        driver.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['http', 'browser'])
    mode = parser.parse_args().mode
    try:
        result = (check_http if mode == 'http' else check_browser)()
    except Exception as exc:
        result = {'error': f'{type(exc).__name__}: {exc}'}
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report, flush=True)
    (OUT / f'{mode}.json').write_text(report, encoding='utf-8')
    # A refusal is a diagnostic result, not a request to retry or bypass it.
    if result.get('blocked') or result.get('error'):
        raise SystemExit(1)
