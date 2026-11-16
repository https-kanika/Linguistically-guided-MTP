from lgmtp.week1 import DEFAULT_W1, verdict

ZS = "Universal head, zero-shot (all small LLMs)"
ENT = "Entropy (Phase 1, 1B draft)"
INM = "In-model head (zero-shot head fine-tuned on 8B dev)"
TXT = "Text-only head, zero-shot (ablation: no distribution input)"


def res(zs, zs_lo, inm, txt=1.3, dist_lo=0.01):
    return dict(l8=[dict(method=ENT, speedup=1.45, ci_lo=1.44, ci_hi=1.46),
                    dict(method=ZS, speedup=zs, ci_lo=zs - 0.01, ci_hi=zs + 0.01, diff_lo=zs_lo),
                    dict(method=INM, speedup=inm, ci_lo=inm - 0.01, ci_hi=inm + 0.01),
                    dict(method=TXT, speedup=txt, ci_lo=txt - 0.01, ci_hi=txt + 0.01)],
                dist_vs_text=dict(diff=0.1, lo=dist_lo, hi=0.2, p_better=0.99))


def test_go_weak_nogo():
    cfg = dict(DEFAULT_W1)
    assert verdict(cfg, res(1.7, 0.2, 1.8))["decision"] == "GO"
    assert verdict(cfg, res(1.5, 0.03, 1.6))["decision"] == "WEAK GO"
    assert verdict(cfg, res(1.5, -0.01, 1.6))["decision"] == "NO GO"          # does not beat Entropy
    assert verdict(cfg, res(1.65, 0.1, 2.5))["decision"] == "NO GO"          # recovery 0.43 < 0.7
    v = verdict(cfg, res(1.7, 0.2, 1.8, dist_lo=-0.02))
    assert v["decision"] == "GO" and not v["distribution_input_helps"]
    assert abs(v["recovery"] - 0.7 / 0.8) < 1e-9


def test_incomplete_without_8b():
    assert verdict(dict(DEFAULT_W1), dict(l8=[]))["decision"] == "INCOMPLETE"
