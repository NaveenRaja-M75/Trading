"""Print public documentation pages (read-only). usage: python research/fetch_docs.py URL [URL ...]"""
import sys
import urllib.request

for url in sys.argv[1:]:
    print(f"\n\n==================== {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/markdown,text/plain,*/*"})
        with urllib.request.urlopen(req, timeout=60) as r:
            print(r.read().decode(errors="replace")[:15000])
    except Exception as e:                                       # noqa: BLE001
        print(f"failed: {e}")
