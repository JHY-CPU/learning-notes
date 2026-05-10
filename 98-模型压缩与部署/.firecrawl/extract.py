import json
import sys

filepath = sys.argv[1]
with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

web = data.get('data', {}).get('web', [])
for r in web:
    print('TITLE:', r.get('title', ''))
    print('URL:', r.get('url', ''))
    md = r.get('markdown', '')
    print('CONTENT:', md[:3000])
    print('=====')
