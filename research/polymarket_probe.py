"""Look at Polymarket's public API shape for the 15-minute 'Bitcoin Up or Down' markets. Read-only."""
import json
import time
import urllib.parse
import urllib.request

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def get(url, params=None):
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(full, headers={"User-Agent": "Mozilla/5.0 research"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300]
    except Exception as e:
        return -1, repr(e)


def short(x, n=1500):
    s = json.dumps(x, default=str) if not isinstance(x, (bytes, str)) else str(x)
    return s[:n]


now = int(time.time())
start = now - now % 900
print("== slug guesses for recent window starts")
found = None
for k in range(1, 6):
    ts = start - 900 * k
    for slug in (f"btc-updown-15m-{ts}", f"bitcoin-up-or-down-15m-{ts}"):
        st, body = get(f"{GAMMA}/events", {"slug": slug})
        print(slug, st, short(body, 300))
        if st == 200 and isinstance(body, list) and body:
            found = found or body[0]

print("\n== search")
for q in ("Bitcoin Up or Down", "btc updown 15m"):
    st, body = get(f"{GAMMA}/public-search", {"q": q, "limit_per_type": 5})
    print(q, st, short(body, 1500))

print("\n== recent closed markets ordered by end date")
st, body = get(f"{GAMMA}/markets", {"limit": 5, "closed": "true", "order": "endDate", "ascending": "false"})
print(st, short([{k: m.get(k) for k in ("slug", "question", "endDate", "outcomes", "outcomePrices", "clobTokenIds")}
                 for m in body] if st == 200 else body, 2500))

if found:
    print("\n== found event keys", list(found.keys()))
    m = found["markets"][0]
    print("market keys", list(m.keys()))
    print(short({k: m.get(k) for k in ("slug", "question", "startDate", "endDate", "eventStartTime", "outcomes",
                                         "outcomePrices", "clobTokenIds", "closed", "umaResolutionStatus",
                                         "feesEnabled", "makerBaseFee", "takerBaseFee", "description")}, 3000))
    tokens = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds")
    for fid in (1,):
        st, body = get(f"{CLOB}/prices-history", {"market": tokens[0], "interval": "1d", "fidelity": fid})
        print("\nprices-history interval=1d fidelity", fid, st, short(body, 1500))
    st, body = get(f"{CLOB}/prices-history", {"market": tokens[0], "startTs": start - 3600 * 3, "endTs": now, "fidelity": 1})
    print("\nprices-history startTs/endTs", st, short(body, 1500))
