import sys
import os; _BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "refund_bound.py")
src = open(_BASE, encoding="utf-8").read(); exec(src[:src.index("rng = ")])
def complete_g(pid, ptxt, tau):
    lo = bisect.bisect_left(words, ptxt); hi = bisect.bisect_left(words, ptxt + "\uffff")
    k = len(pid); cons = []
    for w in words[lo:hi]:
        if w == ptxt: cons.append((w, [])); continue
        t = wtoks(w)
        if len(t) > k and t[:k] == pid: cons.append((w, t[k:]))
    if not cons: return []
    tot = sum(freq[w] for w, _ in cons) or 1
    w, c = max(cons, key=lambda x: freq[x[0]])
    return c if c and freq[w] / tot >= tau and freq[w] >= 2 else []
import builtins
def run(split, tau, lam):
    global complete
    complete = lambda pid, ptxt, ntry=8: complete_g(pid, ptxt, tau)
    globals()["complete"] = complete
    return sim(tr[split], "trie", lam=lam)
rng = np.random.default_rng(0)
base = sim(tr["devtest"], "entropy")
grid = [(tau, lam) for tau in (0.6, 0.8, 0.9, 0.97) for lam in (-0.9, -0.683, -0.5)]
dev = {g: run("dev", *g)[0] for g in grid}
for g, v in sorted(dev.items(), key=lambda x: -x[1])[:5]: print("dev", g, round(v, 3), flush=True)
best = max(dev, key=dev.get)
res = run("devtest", *best)
n = len(res[1]); idx = rng.integers(0, n, (1000, n))
s = res[1][idx].sum(1) / res[2][idx].sum(1); d = s - base[1][idx].sum(1) / base[2][idx].sum(1)
print(f"GATED devtest {best}: {res[0]:.3f} [{np.percentile(s,2.5):.3f},{np.percentile(s,97.5):.3f}] diff vs Entropy [{np.percentile(d,2.5):+.3f},{np.percentile(d,97.5):+.3f}]", flush=True)
