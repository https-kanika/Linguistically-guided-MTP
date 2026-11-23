"""Quick feasibility bound for the 'token-tax refund' drafter on Phase-1 traces (8B target, 1B draft).

Variants (all greedy, lossless, exact teacher-forced simulation like Phase 1):
  Entropy      : Phase-1 rule, every drafted token costs c.
  FreeCont(k)  : same decisions/tokens as Entropy, but drafted word-continuation tokens cost k*c
                 (pure cost upper bound: a completer as accurate as the 1B and nearly free).
  Trie         : 1B drafts only when the current word prefix is empty (word start); otherwise a
                 frequency lexicon completes the word (cost ~0); the 1B's next call processes the
                 completed tokens at once (cost c*r(q)). Entropy rule checked only on 1B tokens.
"""
import json, sys, bisect, math, re
from collections import Counter
import numpy as np

sys.path.insert(0, "src")
from lgmtp.traces import load_traces, attach_word_labels
from lgmtp.simulator import make_verify, load_timing
from lgmtp.policies import entropy_fires
from transformers import AutoTokenizer

D = "data/traces_v2/"
tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
timing = load_timing(D + "timing.json")
C = timing["c"]
V = make_verify(timing)
GMAX = 8
def r_catch(q):           # 1B cost of processing q new tokens in one call, relative to one token
    return V(max(q - 1, 0))  # proxy: same shape as the 8B's measured verify curve (+~1%/token)

tr = {s: load_traces(D + f"traces_{s}.npz")[0] for s in ("dev", "devtest")}
for s in tr: attach_word_labels(tr[s], tok)

# ---- lexicon: word frequencies from dev target outputs (tuning split) + hindi.csv types --------
WRE = re.compile(r"[ऀ-ॿ‌‍]+")
freq = Counter()
for line in open(D + "target_outputs_dev.jsonl", encoding="utf-8"):
    freq.update(WRE.findall(json.loads(line)["out_text"]))
for i, line in enumerate(open("../hindi.csv", encoding="utf-8")):
    if i: w = line.split(",")[0].strip(); freq[w] += 0 if w in freq else 1
words = sorted(freq)
tokcache = {}
def wtoks(w):
    if w not in tokcache: tokcache[w] = tok.encode(" " + w, add_special_tokens=False)
    return tokcache[w]

def complete(prefix_ids, prefix_txt, ntry=8):
    """Most frequent lexicon word extending prefix_txt whose tokenisation starts with prefix_ids.
    Returns the continuation token ids ([] = predict the word is finished)."""
    lo = bisect.bisect_left(words, prefix_txt); hi = bisect.bisect_left(words, prefix_txt + "￿")
    cands = sorted(words[lo:hi], key=lambda w: -freq[w])[:ntry]
    k = len(prefix_ids)
    for w in cands:
        if w == prefix_txt: return []          # most likely: word already complete
        t = wtoks(w)
        if len(t) > k and t[:k] == prefix_ids: return t[k:]
    return []

def word_prefix(trc, p):
    """Token ids and text of the current (unfinished) Devanagari word ending at position p-1."""
    wi = trc.labels["word_initial"]
    if p == 0: return [], ""
    j = p - 1
    while j > 0 and not wi[j]: j -= 1
    ids = [int(x) for x in trc.target_ids[j:p]]
    txt = tok.decode(ids).strip()
    if not txt or not WRE.fullmatch(txt): return ids, ""
    return ids, txt

def sim(traces, mode, lam=-0.683, beta=4.0, kappa=0.0):
    em, co = [], []
    for t in traces:
        n, m, H, wi = t.n, t.match, t.H, t.labels["word_initial"]
        p = 0; E = Cc = 0.0
        while p < n:
            g, cost, stop = 0, 0.0, False
            if mode in ("entropy", "freecont"):
                while g < min(GMAX, n - p) and not stop:
                    pos = p + g
                    cont = mode == "freecont" and not wi[pos]
                    cost += C * (kappa if cont else 1.0)
                    stop = entropy_fires(H[pos], lam, beta); g += 1
                ok = 0
                while ok < g and m[p + ok]: ok += 1
            else:  # trie
                draft_ok, ok, pending = True, 0, 0       # pending = completer tokens the 1B has not seen
                while g < min(GMAX, n - p) and not stop:
                    pos = p + g
                    pid, ptxt = word_prefix(t, pos) if draft_ok else ([], "")
                    comp = complete(pid, ptxt) if ptxt else []
                    if comp:
                        comp = comp[:min(GMAX, n - p) - g]
                        for x in comp:
                            if draft_ok and int(t.target_ids[p + g]) == x: ok += 1
                            else: draft_ok = False
                            g += 1
                        pending += len(comp)
                        continue
                    cost += C * r_catch(1 + pending); pending = 0
                    if draft_ok and m[pos]: ok += 1
                    else: draft_ok = False
                    stop = entropy_fires(H[pos], lam, beta); g += 1
            gained = min(ok + 1, n - p)
            cost += V(g)
            E += gained; Cc += cost; p += gained
        em.append(E); co.append(Cc)
    em, co = np.array(em), np.array(co)
    return em.sum() / co.sum(), em, co

rng = np.random.default_rng(0)
def boot(em, co, B=1000):
    n = len(em); idx = rng.integers(0, n, (B, n))
    s = em[idx].sum(1) / co[idx].sum(1); return np.percentile(s, [2.5, 97.5])
def pboot(a, b, B=1000):
    n = len(a[1]); idx = rng.integers(0, n, (B, n))
    d = a[1][idx].sum(1) / a[2][idx].sum(1) - b[1][idx].sum(1) / b[2][idx].sum(1)
    return np.percentile(d, [2.5, 97.5])

lams = [-1.3, -1.1, -0.9, -0.8, -0.683, -0.6, -0.5, -0.4, -0.3]
out = {}
base = sim(tr["devtest"], "entropy")
out["Entropy (Phase 1)"] = base
for kap in (0.0, 0.05):
    best = max(lams, key=lambda l: sim(tr["dev"], "freecont", lam=l, kappa=kap)[0])
    out[f"FreeCont kappa={kap} (lam={best})"] = sim(tr["devtest"], "freecont", lam=best, kappa=kap)
for l in lams: print("trie dev", l, round(sim(tr["dev"], "trie", lam=l)[0], 3), flush=True)
best = max(lams, key=lambda l: sim(tr["dev"], "trie", lam=l)[0])
out[f"Trie completer (lam={best})"] = sim(tr["devtest"], "trie", lam=best)
for k, v in out.items():
    lo, hi = boot(v[1], v[2]); d = pboot(v, base)
    print(f"{k:38s} {v[0]:.3f} [{lo:.3f},{hi:.3f}]  diff vs Entropy CI [{d[0]:+.3f},{d[1]:+.3f}]")
wi = np.concatenate([t.labels["word_initial"] for t in tr["devtest"]])
print("share of continuation tokens:", 1 - wi.mean(), " lexicon types:", len(words))
