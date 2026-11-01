"""Does the Vercel AI Gateway offer Jev (TypeSafe)? Reads the gateway's public model list. Uses no key."""
import json
import urllib.request

req = urllib.request.Request("https://ai-gateway.vercel.sh/v1/models", headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=60) as r:
    ids = sorted(m["id"] for m in json.loads(r.read()).get("data", []))
print(f"{len(ids)} models on the gateway")
hits = [i for i in ids if "jev" in i.lower() or "typesafe" in i.lower() or "system" in i.lower()]
print("Jev / TypeSafe models:", hits or "NONE")
print("providers:", ", ".join(sorted({i.split('/')[0] for i in ids})))
for i in ids:
    print(" ", i)
