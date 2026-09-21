from pathlib import Path
from playwright.sync_api import sync_playwright, expect

with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    page=browser.new_page(viewport={'width':380,'height':1000})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto('http://127.0.0.1:4173')
    page.wait_for_load_state('networkidle')
    expect(page.locator('#quiet-state')).to_be_hidden()
    expect(page.locator('#lens-summary')).to_contain_text('Blackwell asynchronous MMA')
    expect(page.locator('#lens-summary')).to_contain_text('Next tile copied during current compute')
    expect(page.locator('#lens-summary')).to_contain_text('MMA completion waited on each tile')
    expect(page.locator('#lens-summary')).to_contain_text('TMEM chunks loaded and waited on repeatedly')
    expect(page.locator('#lens-summary')).to_contain_text('Register pressure')
    expect(page.locator('#lens-summary')).to_contain_text('Launch parallelism')
    page.screenshot(path='artifacts/kernel-lens-0.7.0.png',full_page=True)
    row=page.locator('[data-id="review.cuda-mma-cadence"].kernel-axis')
    row.locator('summary').click()
    expect(row).to_contain_text('Next check:')
    expect(row).to_contain_text('Runtime impact is unmeasured')
    expect(row.locator('.source-evidence').first).to_be_visible()
    row.get_by_role('button',name='Show source ↗').click()
    assert page.evaluate("window.kernelLensMessages.some(m=>m.type==='jump'&&m.id==='review.cuda-mma-cadence')")
    page.evaluate("const s=window.kernelLensSampleState;s.stale=true;window.postMessage({type:'state',state:s},'*')")
    expect(row.get_by_role('button',name='Show source ↗')).to_be_disabled()
    page.set_viewport_size({'width':260,'height':950})
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    page.screenshot(path='artifacts/kernel-lens-0.7.0-narrow.png',full_page=True)
    assert not errors, errors
    browser.close()
print('Kernel sidebar passed: distinct visible buckets, source navigation, uncertainty, stale controls, and narrow layout.')
