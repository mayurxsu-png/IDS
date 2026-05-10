"""E2E test with narrow output."""
import requests, re

BASE = 'http://localhost:5000'
presets = requests.get(f'{BASE}/api/presets').json()

for key in presets:
    vals = presets[key]['values'].copy()
    vals['model_choice'] = 'ensemble'
    resp = requests.post(f'{BASE}/predict', data=vals, allow_redirects=False)
    html = resp.text
    m1 = re.search(r'<h2[^>]*class="display-5[^"]*"[^>]*>([^<]+)', html)
    attack = m1.group(1).strip() if m1 else "ERR"
    m2 = re.search(r'Confidence:\s*<strong>([^<]+)', html)
    conf = m2.group(1).strip() if m2 else "?"
    print(f"{key}={attack},{conf}")
