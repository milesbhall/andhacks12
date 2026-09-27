"""Build stance-only replay results (results/replay_<id>.json) for every transcript in
transcripts/catalog.json so the website's Replay tab can show them. No trades, no Tiger rows."""
import json, os, sys
import stance_scorer, pipeline
cat = json.load(open(os.path.join(stance_scorer.TRANSCRIPT_DIR, "catalog.json"), encoding="utf-8"))
store = stance_scorer.load_store()
only = set(sys.argv[1:])
for item in cat:
    rid, speaker = item["id"], item["speaker"]
    if only and rid not in only:
        continue
    out = os.path.join(pipeline.RESULTS_DIR, f"replay_{rid}.json")
    if os.path.isfile(out) or speaker not in store:
        continue
    try:
        results = stance_scorer.replay(speaker, rid)
    except Exception as e:
        print(rid, "failed:", e); continue
    records = [{"speaker": speaker, "statement": r.statement, "stance": r.stance,
                "baseline_mean": r.baseline_mean, "baseline_stdev": r.baseline_stdev,
                "z": round(r.z, 2), "direction": r.direction, "summary": r.summary,
                "matches": [], "trades": []} for r in results]
    pipeline.save(f"replay_{rid}", records)
    print(rid, len(records), "answers,", sum(r["direction"] != "IN LINE" for r in records), "surprises")
