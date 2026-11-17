"""Diagnose DIA's existing Selenium search in CI without writing RAW data."""
import json
import traceback
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dia_diagnostics'
OUT.mkdir(exist_ok=True)
nb = nbformat.read(ROOT / 'scrapers/dia.ipynb', as_version=4)
namespace = {}
# Reuse only the configuration, imports and helper definitions; never run the
# notebook's collection or save cells.
markers = ['HOME_URL =', 'def start_driver(', 'def find_search_box(',
           'def load_all_cards(', 'def extract_products(']
for marker in markers:
    candidates = [c.source for c in nb.cells if c.cell_type == 'code' and marker in c.source]
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one helper cell for {marker}')
    exec(compile(candidates[0], f'dia:{marker}', 'exec'), namespace)

webdriver = namespace['webdriver']
Options = namespace['Options']
By = namespace['By']
Keys = namespace['Keys']
WebDriverWait = namespace['WebDriverWait']
EC = namespace['EC']
options = Options()
for arg in ('--headless=new', '--no-sandbox', '--disable-dev-shm-usage', '--window-size=1920,1080'):
    options.add_argument(arg)

driver = None
stage = 'start_browser'

def capture(label):
    if driver is None:
        return
    print(f'\n--- {label} ---', flush=True)
    try:
        body = driver.find_element(By.TAG_NAME, 'body').text
        info = {'url': driver.current_url, 'title': driver.title,
                'body_excerpt': body[:3000],
                'cards': len(driver.find_elements(By.CSS_SELECTOR, namespace['CARD_SELECTOR'])),
                'containers': len(driver.find_elements(By.CSS_SELECTOR, namespace['SCROLL_SELECTOR'])),
                'inputs': driver.execute_script("return Array.from(document.querySelectorAll('input')).map(e => ({type:e.type, placeholder:e.placeholder, ariaLabel:e.getAttribute('aria-label'), testId:e.getAttribute('data-test-id')}));")}
        print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
        (OUT / f'{label}.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        print(f'Page inspection failed: {exc}', flush=True)
    try:
        driver.save_screenshot(str(OUT / f'{label}.png'))
    except Exception as exc:
        print(f'Screenshot failed: {exc}', flush=True)
    try:
        (OUT / f'{label}.html').write_text(driver.page_source, encoding='utf-8')
    except Exception as exc:
        print(f'HTML capture failed: {exc}', flush=True)

try:
    driver = webdriver.Chrome(options=options)
    driver.set_page_load_timeout(45)
    stage = 'home'
    namespace['go_to_home'](driver)
    capture('01_home')
    namespace['assert_no_security_challenge'](driver)
    stage = 'find_search_box'
    box = namespace['find_search_box'](driver)
    stage = 'submit_search'
    box.click()
    box.send_keys(Keys.CONTROL, 'a')
    box.send_keys(Keys.BACKSPACE)
    box.send_keys('leche')
    box.send_keys(Keys.ENTER)
    stage = 'wait_for_container'
    WebDriverWait(driver, namespace['WAIT']).until(EC.presence_of_element_located((By.CSS_SELECTOR, namespace['SCROLL_SELECTOR'])))
    stage = 'wait_for_cards'
    WebDriverWait(driver, namespace['WAIT']).until(lambda d: len(d.find_elements(By.CSS_SELECTOR, namespace['CARD_SELECTOR'])) > 0)
    capture('02_results')
    namespace['assert_no_security_challenge'](driver)
    stage = 'extract_visible_products'
    frame = namespace['extract_products'](driver, 'leche')
    print(f'Visible products extracted: {len(frame)}', flush=True)
    if frame.empty:
        raise RuntimeError('Cards appeared but no products were extracted.')
    print('PASS: DIA search and visible product extraction.', flush=True)
except Exception:
    print(f'FAILED_STAGE: {stage}', flush=True)
    capture('failure')
    (OUT / 'error.txt').write_text(f'FAILED_STAGE: {stage}\n{traceback.format_exc()}', encoding='utf-8')
    raise
finally:
    if driver is not None:
        driver.quit()
