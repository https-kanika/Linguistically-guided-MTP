import sys, runpy
import os; _BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "refund_bound.py")
src = open(_BASE, encoding="utf-8").read()
src = src[:src.index("rng = ")]
exec(src)
# at each continuation position whose word prefix is Devanagari: completer vs 1B chain
stats = dict(n=0, trie_full=0, trie_first=0, trie_empty=0, b1_full=0, b1_first=0)
for t in tr["devtest"]:
    wi = t.labels["word_initial"]; n = t.n
    for pos in range(1, n):
        if wi[pos]: continue
        pid, ptxt = word_prefix(t, pos)
        if not ptxt: continue
        end = pos
        while end < n and not wi[end]: end += 1
        truth = [int(x) for x in t.target_ids[pos:end]]
        comp = complete(pid, ptxt)
        stats["n"] += 1
        stats["trie_empty"] += comp == []
        stats["trie_first"] += bool(comp) and comp[0] == truth[0]
        stats["trie_full"] += comp == truth
        stats["b1_first"] += bool(t.match[pos])
        stats["b1_full"] += bool(t.match[pos:end].all())
print({k: (v if k == "n" else round(v / stats["n"], 3)) for k, v in stats.items()})
