import sys
import os; _BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "refund_bound.py")
src = open(_BASE, encoding="utf-8").read(); exec(src[:src.index("rng = ")])
from collections import defaultdict
recs = {s: {json.loads(l)["id"]: json.loads(l) for l in open(D + f"target_outputs_{s}.jsonl", encoding="utf-8")} for s in ("dev", "devtest")}
# bigram from dev outputs
big = defaultdict(Counter)
for r in recs["dev"].values():
    ws = WRE.findall(r["out_text"])
    for a, b in zip(["<s>"] + ws, ws): big[a][b] += 1

def complete_ctx(pid, ptxt, prev, cache, bag, use):
    k = len(pid)
    lo = bisect.bisect_left(words, ptxt); hi = bisect.bisect_left(words, ptxt + "\uffff")
    def score(w):
        s = math.log(freq[w] + 1)
        if "bigram" in use: s += 3 * math.log(1 + big[prev][w])
        if "cache" in use and w in cache: s += 5
        if "bag" in use and w in bag: s += 10
        return s
    cands = list(words[lo:hi])
    if "bag" in use:
        cands += [w for w in bag if w.startswith(ptxt) and w not in freq]
    cands = sorted(set(cands), key=lambda w: -score(w))[:8]
    for w in cands:
        if w == ptxt: return []
        t = wtoks(w)
        if len(t) > k and t[:k] == pid: return t[k:]
    return []

def evaluate(use):
    nfull = n = 0
    for t in tr["devtest"]:
        wi = t.labels["word_initial"]; r = recs["devtest"][t.id]
        bag = set(WRE.findall(r["ref"])) if "bag" in use else set()
        for pos in range(1, t.n):
            if wi[pos]: continue
            pid, ptxt = word_prefix(t, pos)
            if not ptxt: continue
            j = pos - 1
            while j > 0 and not wi[j]: j -= 1
            before = WRE.findall(tok.decode([int(x) for x in t.target_ids[:j]]))
            prev = before[-1] if before else "<s>"
            end = pos
            while end < t.n and not wi[end]: end += 1
            comp = complete_ctx(pid, ptxt, prev, set(before), bag, use)
            n += 1; nfull += comp == [int(x) for x in t.target_ids[pos:end]]
    return nfull / n
for use in [(), ("cache",), ("bigram",), ("cache", "bigram"), ("bag",), ("bag", "cache", "bigram")]:
    print(use or "prefix only", round(evaluate(use), 3), flush=True)
