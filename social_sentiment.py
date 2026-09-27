"""
social_sentiment.py
===================
What the crowd expects from the Fed, from Bluesky and Reddit, scored on the same
-1 (dovish: cuts coming) to +1 (hawkish: hikes / higher for longer) scale as the speakers.

The desk compares a speaker's surprise with the crowd: a hawkish surprise when the
crowd is dovish is a bigger repricing than one the crowd already expected.

  python social_sentiment.py                 # fetch, score, print
  python social_sentiment.py --query "Warsh rate hike"

Bluesky: public AppView search, no key. Reddit: official API with redditapi.txt
('client_id:client_secret' of a script app), falling back to the public JSON.
Gemini (gemapi.txt) scores the posts in one batched call.
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "social_cache.json")
UA = {"User-Agent": "MarketPulse/1.0 (andhacks hackathon research; contact omofkmg@gmail.com)"}
DEFAULT_QUERIES = ["Fed rate", "Warsh", "FOMC"]
REDDIT_SUBS = "economics+investing+stocks+wallstreetbets+finance+Kalshi+Polymarket"
PROMPT = """You rate social media posts about the US Federal Reserve.
For each post, give "stance" from -1 to +1 for what the author EXPECTS or WANTS the Fed to do:
-1 = clearly dovish (rate cuts coming / should cut), 0 = neutral or mixed, +1 = clearly hawkish
(hikes coming / higher for longer / inflation fight). Set "relevant": false if the post is not
about Fed policy, interest rates or inflation. Return JSON
{"scores": [{"id": int, "stance": float, "relevant": bool, "summary": "<=10 words"}]}.
"""


def bluesky(query: str, limit: int = 40) -> list:
    r = requests.get("https://api.bsky.app/xrpc/app.bsky.feed.searchPosts",
                     params={"q": query, "limit": limit, "sort": "latest", "lang": "en"},
                     headers=UA, timeout=20)
    r.raise_for_status()
    out = []
    for p in r.json().get("posts", []):
        rec, author = p.get("record") or {}, p.get("author") or {}
        uri = p.get("uri", "")
        rkey = uri.rsplit("/", 1)[-1]
        out.append({"source": "Bluesky", "text": (rec.get("text") or "").strip(),
                    "author": "@" + author.get("handle", ""), "time": rec.get("createdAt"),
                    "likes": p.get("likeCount", 0),
                    "url": f"https://bsky.app/profile/{author.get('handle', '')}/post/{rkey}"})
    return out


_reddit_token = {"value": "", "expires": 0.0}


def _reddit_credentials():
    """redditapi.txt = 'client_id:client_secret' from a 'script' app at reddit.com/prefs/apps."""
    raw = os.environ.get("REDDIT_API", "")
    path = os.path.join(SCRIPT_DIR, "redditapi.txt")
    if not raw and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
    return tuple(raw.split(":", 1)) if ":" in raw else None


def _reddit_headers() -> dict:
    creds = _reddit_credentials()
    if not creds:
        return {}
    if time.time() > _reddit_token["expires"]:
        for host in ("https://www.reddit.com", "https://ssl.reddit.com"):
            try:
                r = requests.post(f"{host}/api/v1/access_token", auth=creds, headers=UA, timeout=20,
                                  data={"grant_type": "client_credentials"})
                r.raise_for_status()
                tok = r.json()
                _reddit_token.update(value=tok["access_token"], expires=time.time() + tok.get("expires_in", 3600) - 60)
                break
            except (requests.RequestException, KeyError, ValueError):
                continue
    return {"Authorization": f"bearer {_reddit_token['value']}"} if _reddit_token["value"] else {}


def reddit(query: str, limit: int = 40) -> list:
    """Reddit's official API (OAuth app-only) when redditapi.txt exists, else the public JSON."""
    auth = _reddit_headers()
    base = "https://oauth.reddit.com" if auth else "https://www.reddit.com"
    r = requests.get(f"{base}/r/{REDDIT_SUBS}/search" + ("" if auth else ".json"),
                     params={"q": query, "restrict_sr": "1", "sort": "new", "t": "week", "limit": limit, "raw_json": 1},
                     headers={**UA, **auth}, timeout=20)
    r.raise_for_status()
    out = []
    for child in r.json().get("data", {}).get("children", []):
        d = child.get("data", {})
        text = (d.get("title", "") + ". " + (d.get("selftext") or "")[:400]).strip()
        out.append({"source": "Reddit", "text": text, "author": "r/" + d.get("subreddit", ""),
                    "time": datetime.fromtimestamp(d.get("created_utc", 0), timezone.utc).isoformat(),
                    "likes": d.get("score", 0), "url": "https://www.reddit.com" + d.get("permalink", "")})
    return out


def collect(queries=DEFAULT_QUERIES) -> list:
    posts, seen = [], set()
    for q in queries:
        for fn in (bluesky, reddit):
            try:
                batch = fn(q)
            except (requests.RequestException, ValueError) as e:
                print(f"  {fn.__name__} '{q}' skipped: {str(e)[:80]}")
                continue
            for p in batch:
                key = p["url"]
                if p["text"] and key not in seen and len(p["text"].split()) >= 5:
                    seen.add(key)
                    posts.append(p)
    return posts


def score(posts: list) -> list:
    import stance_scorer
    for start in range(0, len(posts), 40):
        batch = posts[start:start + 40]
        data = stance_scorer._gemini_json(PROMPT + json.dumps(
            [{"id": i, "text": p["text"][:600]} for i, p in enumerate(batch)], ensure_ascii=False))
        by_id = {int(s["id"]): s for s in data.get("scores", []) if "id" in s}
        for i, p in enumerate(batch):
            s = by_id.get(i, {})
            p["relevant"] = bool(s.get("relevant", False))
            p["stance"] = max(-1.0, min(1.0, float(s.get("stance", 0) or 0)))
            p["summary"] = s.get("summary", "")
    return posts


def summarize(posts: list) -> dict:
    rel = [p for p in posts if p.get("relevant")]
    by_source = {}
    for src in ("Bluesky", "Reddit"):
        xs = [p["stance"] for p in rel if p["source"] == src]
        by_source[src] = {"n": len(xs), "mean": round(statistics.mean(xs), 3) if xs else None}
    xs = [p["stance"] for p in rel]
    mean = statistics.mean(xs) if xs else 0.0
    lean = "HAWKISH" if mean > 0.15 else "DOVISH" if mean < -0.15 else "MIXED"
    rel.sort(key=lambda p: (abs(p["stance"]), p.get("likes") or 0), reverse=True)
    return {"updated_at": datetime.now(timezone.utc).isoformat(), "n": len(rel), "mean": round(mean, 3),
            "lean": lean, "by_source": by_source,
            "posts": [{k: p.get(k) for k in ("source", "author", "text", "stance", "summary", "url", "time", "likes")}
                      | {"text": p["text"][:280]} for p in rel[:20]]}


def latest(max_age: int = 600, queries=DEFAULT_QUERIES) -> dict:
    """Cached for 10 minutes (one Gemini call per refresh)."""
    try:
        if time.time() - os.path.getmtime(CACHE_PATH) < max_age:
            with open(CACHE_PATH, encoding="utf-8") as f:
                return json.load(f)
    except (OSError, ValueError):
        pass
    posts = collect(queries)
    if not posts:
        return {}
    data = summarize(score(posts))
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    return data


def crowd_note(direction: str, data: dict = None) -> str:
    """How a speaker surprise compares with the crowd, for the live desk."""
    data = data or {}
    if not data.get("n") or direction not in ("HAWKISH", "DOVISH"):
        return ""
    against = (direction == "HAWKISH" and data["lean"] == "DOVISH") or (direction == "DOVISH" and data["lean"] == "HAWKISH")
    return (f"crowd leans {data['lean'].lower()} ({data['mean']:+.2f}, {data['n']} posts)"
            + (" — surprise runs AGAINST the crowd" if against else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bluesky + Reddit Fed sentiment")
    ap.add_argument("--query", nargs="+")
    args = ap.parse_args()
    d = latest(max_age=0, queries=args.query or DEFAULT_QUERIES)
    print(f"{d.get('n', 0)} relevant posts · crowd {d.get('lean')} {d.get('mean', 0):+.2f} · {d.get('by_source')}")
    for p in d.get("posts", [])[:8]:
        print(f"  {p['stance']:+.2f} {p['source']:8} {p['text'][:100]}")
