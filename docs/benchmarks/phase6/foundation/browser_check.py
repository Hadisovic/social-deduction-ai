import json
from pathlib import Path
from playwright.sync_api import sync_playwright

output = Path(__file__).with_name('phase6-browser-results.json')
results = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    for width in (1440, 390):
        page = browser.new_page(viewport={'width': width, 'height': 900}, reduced_motion='reduce')
        errors, bad = [], []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('console', lambda m: errors.append(m.text) if m.type == 'error' else None)
        page.on('response', lambda r: bad.append(r.url) if r.status >= 400 else None)
        page.goto('http://127.0.0.1:4186/social-deduction-ai/?debugNav=1', wait_until='networkidle')
        page.wait_for_function('window.__CREWMATE_DEBUG__ !== undefined')
        assert page.locator('#crewHadi').get_attribute('data-color') == 'red'
        assert page.locator('#crewMasa').get_attribute('data-color') == 'black'
        assert page.locator('#phaseCompletion').inner_text() == '6 / 7'
        page.locator('[data-milestone="mission12"]').last.click()
        page.wait_for_function("document.querySelector('#panelTitle')?.textContent === 'Phase 6 Foundations'", timeout=90000)
        page.get_by_text('OPEN THE MISSION LOG', exact=True).click()
        assert page.get_by_text('6.0 — baseline and vision · complete', exact=True).count() == 1
        assert page.get_by_text('6.4 — multi-agent learning · planned', exact=True).count() == 1
        image = page.locator('#terminalContent img').first
        image.wait_for(state='visible')
        page.wait_for_function("[...document.querySelectorAll('#terminalContent img')].every(i => i.complete && i.naturalWidth > 0)")
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path=str(output.with_name(f'phase6-{width}.png')), full_page=True)
        page.locator('#closeMilestone').click()
        page.evaluate("showMilestone(MILESTONES.find(m => m.id === 'mission11'))")
        assert page.locator('#terminalContent').inner_text().find('88.4% / 90.4%') >= 0
        assert page.locator('#terminalContent .panel-chart').count() == 12
        assert not errors and not bad, (errors, bad)
        results.append(dict(width=width, mission12='passed', phase5_preserved=True,
                            collaborators='red/black', base_path='passed', errors=errors, failed_assets=bad))
        page.close()
    browser.close()
output.write_text(json.dumps(results, indent=2))
print(json.dumps(results))
