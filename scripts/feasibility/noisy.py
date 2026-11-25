import sys
import os; _BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "refund_bound.py")
src = open(_BASE, encoding="utf-8").read(); exec(src[:src.index("rng = ")])
def sim_noisy(traces, rho, lam, beta=4.0, kappa=0.05, seed=0):
    rg = np.random.default_rng(seed); E = Cc = 0.0
    for t in traces:
        n, m, H, wi = t.n, t.match, t.H, t.labels["word_initial"]
        keep = rg.random(n) < rho
        mm = m & (wi | keep)            # continuation tokens right only if the 1B is right AND w.p. rho
        p = 0
        while p < n:
            g, cost, stop = 0, 0.0, False
            while g < min(GMAX, n - p) and not stop:
                pos = p + g; cost += C * (kappa if not wi[pos] else 1.0)
                stop = entropy_fires(H[pos], lam, beta); g += 1
            ok = 0
            while ok < g and mm[p + ok]: ok += 1
            gained = min(ok + 1, n - p); cost += V(g); E += gained; Cc += cost; p += gained
    return E / Cc
lams = [-1.3, -0.9, -0.683, -0.4]
for rho in (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
    best = max(lams, key=lambda l: sim_noisy(tr["dev"], rho, l))
    print(f"rho={rho}: devtest speed-up {sim_noisy(tr['devtest'], rho, best):.3f} (lam={best})", flush=True)
