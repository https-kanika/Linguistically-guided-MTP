"""Tokenizer-independent inputs and unit targets for the universal head (PLAN_phase2_indic_universal_mtp, W1).

A *record* is one generated output of any LLM:
    dict(id, prompt_text, out_ids [n] (incl. a final eos id if the model stopped), eos_end: bool,
         topk_ids [n, K], topk_logp [n, K])          # row t = target distribution q_t over out_ids[t]

Index convention (as in Phase 1): position p (0 <= p < n) means "out_ids[:p] are committed". At p the
head may read the committed text and the distributions q_t for t < p (each committed token was
produced by a target pass that also gave q_t). It must predict the text that follows. Nothing here
reads q_t for t >= p, so features at p are causal by construction (tested in tests/test_uni_features.py).

Everything the head sees is text: characters (``unichar.CharCodec``) and, for every candidate token of
the last M distributions, its *string* relative to the committed text plus its log-probability. The
LLM's token ids never reach the head, which is why one head can serve any LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boundaries import token_to_bytes
from .unichar import EOS, PAD, PEND, SEP, CharCodec, UnitVocab, continues_unit, segment_units

K_CAND, S_CHARS, M_LAGS, K_HEADS, L_PROMPT, L_OUT = 20, 8, 3, 3, 160, 96
F_STATIC = 7          # logp/10, p, prefix mass, len/8, is_eos, incomplete, valid
F_DYN = 3             # delta/16, consistent with committed text, extends beyond it
N_FEAT = F_STATIC + F_DYN


# --------------------------------------------------------------------------- #
# bytes / text helpers
# --------------------------------------------------------------------------- #

def utf8_offsets(b: bytes) -> tuple[str, np.ndarray]:
    """Decode ``b``; return (text, char byte offsets of length len(text)+1).

    Invalid bytes become U+FFFD (one byte each). A trailing incomplete char is left out, so
    ``offsets[-1]`` is the end of the last complete char (can be < len(b))."""
    try:
        text = b.decode("utf-8")
        a = np.frombuffer(b, np.uint8)
        starts = np.flatnonzero((a & 0xC0) != 0x80)
        return text, np.append(starts, len(b)).astype(np.int64)
    except UnicodeDecodeError:
        pass
    chars, offs, i, n = [], [], 0, len(b)
    while i < n:
        x = b[i]
        L = 1 if x < 0x80 else 2 if 0xC2 <= x <= 0xDF else 3 if 0xE0 <= x <= 0xEF else 4 if 0xF0 <= x <= 0xF4 else 0
        if L == 0 or (0x80 <= x <= 0xBF):
            chars.append("�"); offs.append(i); i += 1
            continue
        if i + L > n:
            if all(0x80 <= y <= 0xBF for y in b[i + 1:]):
                break                                  # trailing incomplete char
            chars.append("�"); offs.append(i); i += 1
            continue
        try:
            chars.append(b[i:i + L].decode("utf-8")); offs.append(i); i += L
        except UnicodeDecodeError:
            chars.append("�"); offs.append(i); i += 1
    return "".join(chars), np.array(offs + [i], np.int64)   # i = end of the last complete char


def special_id_set(tok) -> set[int]:
    from .vocab import special_ids
    return special_ids(tok)


class TokTable:
    """Static per-tokenizer tables: token bytes and each token's chars when read on its own."""

    def __init__(self, tok, eos_ids, vocab_size: int | None = None, codec: CharCodec | None = None,
                 S: int = S_CHARS):
        codec = codec or CharCodec()
        n = len(tok)
        V = max(n, int(vocab_size or 0))
        self.size = V
        self.eos_ids = sorted(set(int(i) for i in eos_ids))
        spec = special_id_set(tok) | set(self.eos_ids)
        toks = tok.convert_ids_to_tokens(list(range(n)))
        self.bytes: list[bytes] = [b""] * V
        self.special = np.ones(V, bool)
        for i, t in enumerate(toks):
            if i in spec or t is None:
                continue
            self.bytes[i] = token_to_bytes(t)
            self.special[i] = False
        self.is_eos = np.zeros(V, bool)
        self.is_eos[[i for i in self.eos_ids if i < V]] = True
        self.chars = np.zeros((V, S), np.int16)
        self.scr = np.zeros((V, S), np.int8)
        self.clen = np.zeros(V, np.int8)
        self.cont_start = np.zeros(V, bool)
        self.incomplete = np.zeros(V, bool)
        for i in range(V):
            if self.is_eos[i]:
                self.chars[i, 0], self.clen[i] = EOS, 1
                continue
            b = self.bytes[i]
            if not b:
                continue
            if 0x80 <= b[0] <= 0xBF:
                self.cont_start[i] = True
                continue
            text, offs = utf8_offsets(b)
            self.incomplete[i] = offs[-1] < len(b)
            u, s = codec.encode(text[:S])
            self.chars[i, :len(u)], self.scr[i, :len(u)], self.clen[i] = u, s, len(u)
        self.valid = ~(self.special & ~self.is_eos) & ~self.cont_start

    def seq_bytes(self, ids) -> list[bytes]:
        return [self.bytes[int(i)] if int(i) < self.size else b"" for i in ids]


def make_encoder(tok):
    """``encode(list[str]) -> list[list[int]]`` without special tokens and without a SentencePiece dummy
    prefix space (detected once; then the text is encoded after a newline that is stripped again)."""
    def raw(texts):
        return tok(list(texts), add_special_tokens=False)["input_ids"]

    probe = raw(["x"])[0]
    pb = b"".join(token_to_bytes(t) for t in tok.convert_ids_to_tokens(probe))
    if pb == b"x":
        return raw

    def strip_nl(ids):
        acc = b""
        for j, t in enumerate(tok.convert_ids_to_tokens(ids)):
            acc += token_to_bytes(t)
            if acc.strip(b" ") == b"\n":
                return ids[j + 1:]
            if len(acc) > 2:
                break
        return None

    def enc(texts):
        texts = list(texts)
        out = raw(["\n" + t for t in texts])
        res = []
        for t, ids in zip(texts, out):
            s = strip_nl(ids)
            res.append(s if s is not None else raw([t])[0])
        return res
    return enc


# --------------------------------------------------------------------------- #
# per-sequence positions and unit targets
# --------------------------------------------------------------------------- #

@dataclass
class SeqText:
    """Committed-text bookkeeping of one output sequence."""
    out_bytes: bytes
    text: str
    coff: np.ndarray          # char byte offsets, len(text)+1
    b_tok: np.ndarray         # committed bytes after p tokens, p = 0..n
    c: np.ndarray             # complete chars committed after p tokens
    pend: np.ndarray          # bytes of an incomplete char committed after p tokens
    unit_start: np.ndarray    # char start of each unit
    unit_of_char: np.ndarray
    unit_strs: list = field(default_factory=list)


def seq_text(table: TokTable, out_ids) -> SeqText:
    bl = table.seq_bytes(out_ids)
    b_tok = np.zeros(len(bl) + 1, np.int64)
    b_tok[1:] = np.cumsum([len(b) for b in bl])
    out_bytes = b"".join(bl)
    return text_positions(out_bytes, b_tok)


def text_positions(out_bytes: bytes, byte_cuts: np.ndarray) -> SeqText:
    text, coff = utf8_offsets(out_bytes)
    c = np.searchsorted(coff[1:], byte_cuts, side="right")
    pend = byte_cuts - coff[c]
    spans = segment_units(text)
    us = np.array([a for a, _ in spans], np.int64)
    uoc = np.repeat(np.arange(len(spans)), [b - a for a, b in spans]) if spans else np.zeros(0, np.int64)
    return SeqText(out_bytes, text, coff, byte_cuts.astype(np.int64), c, pend, us, uoc,
                   [text[a:b] for a, b in spans])


def partial_start(st: SeqText, i: int) -> int:
    """Char index where the unit being written at cut i starts (inference rule, needs no future text)."""
    cp, pl = int(st.c[i]), int(st.pend[i])
    if cp == 0:
        return 0
    if pl and not continues_unit(st.text[cp - 1]):
        return cp
    return int(st.unit_start[st.unit_of_char[cp - 1]])


def position_targets(st: SeqText, vocab: UnitVocab, eos_end: bool, k: int = K_HEADS, cuts=None):
    """For each cut i (default: all but the last): partial-unit start byte, head-1 prefix range
    [lo, hi) and the k unit targets (-100 = ignore). Head 1 targets the full unit being written (the
    committed part is a prefix of it); heads 2..k target the following units, then <eos>."""
    cuts = range(len(st.b_tok) - 1) if cuts is None else cuts
    uid = [vocab.id(u) for u in st.unit_strs]
    start2unit = {int(a): j for j, a in enumerate(st.unit_start)}
    nU = len(uid)
    ps_b, lo, hi, tg = [], [], [], []
    for i in cuts:
        ps = partial_start(st, i)
        pb = int(st.coff[ps])
        part = st.out_bytes[pb:int(st.b_tok[i])]
        a, b = vocab.prefix_range(part)
        j = start2unit.get(ps)
        row = [-100] * k
        if j is not None:
            for h in range(k):
                jj = j + h
                if jj < nU:
                    row[h] = uid[jj]
                elif jj == nU and eos_end:
                    row[h] = vocab.eos_id
                    break
                else:
                    break
            if not (a <= row[0] < b):
                row[0] = -100
        ps_b.append(pb); lo.append(a); hi.append(b); tg.append(row)
    return (np.array(ps_b, np.int64), np.array(lo, np.int32), np.array(hi, np.int32),
            np.array(tg, np.int32).reshape(-1, k))


# --------------------------------------------------------------------------- #
# candidates (the target's top-K, re-expressed as strings)
# --------------------------------------------------------------------------- #

def candidate_arrays(st: SeqText, table: TokTable, topk_ids, topk_logp, text_u, codec: CharCodec,
                     eos_end: bool, S: int = S_CHARS):
    """Per token t: chars of each candidate relative to the committed text before t, its length,
    common-prefix length with the actual text from the anchor (only compared up to the committed end
    when used), static features, validity."""
    n, K = topk_ids.shape
    ids = np.asarray(topk_ids, np.int64)
    ids = np.where(ids < table.size, ids, 0)
    cu = table.chars[ids].copy()                 # (n, K, S)
    cs = table.scr[ids].copy()
    cl = table.clen[ids].astype(np.int16)
    inc = table.incomplete[ids].copy()
    val = table.valid[ids].copy()
    val &= np.asarray(topk_ids) < table.size
    for t in np.flatnonzero(st.pend[:n] > 0):    # tokens that start inside a char: decode pend + bytes
        pb = st.out_bytes[int(st.coff[st.c[t]]):int(st.b_tok[t])]
        for j in range(K):
            v = int(ids[t, j])
            cu[t, j], cs[t, j], cl[t, j], inc[t, j], val[t, j] = 0, 0, 0, False, False
            if table.is_eos[v] or table.special[v]:
                continue
            txt, offs = utf8_offsets(pb + table.bytes[v])
            if not txt or txt[0] == "�":
                continue
            u, s = codec.encode(txt[:S])
            cu[t, j, :len(u)], cs[t, j, :len(u)], cl[t, j] = u, s, len(u)
            inc[t, j], val[t, j] = offs[-1] < len(pb) + len(table.bytes[v]), True
    # common prefix with the real text from the anchor char c[t]
    ext = np.concatenate([text_u, np.array([EOS] if eos_end else [], np.int16), np.zeros(S, np.int16)])
    W = ext[st.c[:n, None] + np.arange(S)[None]]                    # (n, S)
    sidx = np.arange(S)
    eq = (cu == W[:, None, :]) & (sidx[None, None] < cl[..., None])
    lmatch = np.cumprod(eq, axis=2).sum(2).astype(np.int8)
    # char-prefix mass: sum of probabilities of the candidates whose string starts with this one
    p = np.exp(np.clip(np.asarray(topk_logp, np.float32), -30, 0)) * val
    same = cu[:, None, :, :] == cu[:, :, None, :]                    # [n, i, j, s]
    within = sidx[None, None, None] < cl[:, :, None, None]           # s < len_i
    starts = np.all(same | ~within, axis=3) & (cl[:, None, :] >= cl[:, :, None]) & val[:, None, :]
    pmass = (starts * p[:, None, :]).sum(2)
    lp = np.clip(np.asarray(topk_logp, np.float32), -30, 0)
    f = np.stack([lp / 10, p, pmass, cl / 8, table.is_eos[ids], inc, val], -1).astype(np.float16)
    return cu, cs, cl.astype(np.int8), lmatch, f, val


# --------------------------------------------------------------------------- #
# FeatureSet: flat arrays for many sequences, vectorised batch building
# --------------------------------------------------------------------------- #

ARRAYS = ("prompt_u", "prompt_s", "text_u", "text_s", "seq_char0", "seq_tok0", "pos_seq", "pos_c", "pos_pend",
          "pos_tok", "tgt", "lo", "hi", "cand_u", "cand_s", "cand_len", "cand_match", "cand_f", "cand_valid")


@dataclass
class FeatureSet:
    prompt_u: np.ndarray
    prompt_s: np.ndarray
    text_u: np.ndarray
    text_s: np.ndarray
    seq_char0: np.ndarray
    seq_tok0: np.ndarray
    pos_seq: np.ndarray
    pos_c: np.ndarray
    pos_pend: np.ndarray
    pos_tok: np.ndarray
    tgt: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    cand_u: np.ndarray = None
    cand_s: np.ndarray = None
    cand_len: np.ndarray = None
    cand_match: np.ndarray = None
    cand_f: np.ndarray = None
    cand_valid: np.ndarray = None
    seq_ids: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def has_cands(self) -> bool:
        return self.cand_u is not None

    @property
    def n_pos(self) -> int:
        return len(self.pos_seq)

    def trainable(self) -> np.ndarray:
        return np.flatnonzero((self.tgt != -100).any(1))

    def split_by_seq(self, frac: float, seed: int = 0):
        """Position indices (train, val) with ``frac`` of the sequences in val."""
        rng = np.random.default_rng(seed)
        n_seq = len(self.prompt_u)
        val_seq = rng.random(n_seq) < frac
        tr = self.trainable()
        m = val_seq[self.pos_seq[tr]]
        return tr[~m], tr[m]

    def save(self, path) -> None:
        import json
        d = {k: getattr(self, k) for k in ARRAYS if getattr(self, k) is not None}
        np.savez(path, **d, _seq_ids=np.array(self.seq_ids, np.int64),
                 _meta=np.array(json.dumps(self.meta, ensure_ascii=False)))

    @classmethod
    def load(cls, path) -> "FeatureSet":
        import json
        z = np.load(path, allow_pickle=False)
        kw = {k: z[k] for k in ARRAYS if k in z.files}
        return cls(**kw, seq_ids=z["_seq_ids"].tolist(), meta=json.loads(str(z["_meta"])))


def concat(sets: list[FeatureSet]) -> FeatureSet:
    if len(sets) == 1:
        return sets[0]
    hc = {s.has_cands for s in sets}
    if len(hc) != 1:
        raise ValueError("cannot concatenate feature sets with and without candidates")
    out = {k: [] for k in ARRAYS}
    ns = nc = nt = 0
    for s in sets:
        for k in ("prompt_u", "prompt_s", "text_u", "text_s", "pos_pend", "tgt", "lo", "hi"):
            out[k].append(getattr(s, k))
        out["seq_char0"].append(s.seq_char0 + nc)
        out["seq_tok0"].append(s.seq_tok0 + nt)
        out["pos_seq"].append(s.pos_seq + ns)
        out["pos_c"].append(s.pos_c + nc)
        out["pos_tok"].append(np.where(s.pos_tok >= 0, s.pos_tok + nt, -1))
        if s.has_cands:
            for k in ("cand_u", "cand_s", "cand_len", "cand_match", "cand_f", "cand_valid"):
                out[k].append(getattr(s, k))
        ns += len(s.prompt_u)
        nc += len(s.text_u)
        nt += len(s.cand_u) if s.has_cands else 0
    kw = {k: (np.concatenate(v) if v else None) for k, v in out.items()}
    return FeatureSet(**kw, seq_ids=sum((s.seq_ids for s in sets), []),
                      meta=dict(parts=[s.meta for s in sets]))


def _prompt_arrays(texts, codec: CharCodec, L: int):
    pu = np.zeros((len(texts), L), np.int16)
    ps = np.zeros((len(texts), L), np.int8)
    for i, t in enumerate(texts):
        u, s = codec.encode((t or "")[-L:])
        if len(u):
            pu[i, -len(u):], ps[i, -len(u):] = u, s
    return pu, ps


def featurize_records(records: list[dict], table: TokTable, vocab: UnitVocab, codec: CharCodec | None = None,
                      k: int = K_HEADS, with_cands: bool = True, meta: dict | None = None) -> FeatureSet:
    """One example per position p = 0..n-1 of every record (positions are global token indices)."""
    codec = codec or CharCodec()
    text_u, text_s, cand = [], [], {k_: [] for k_ in ("cu", "cs", "cl", "lm", "f", "v")}
    pos_c, pos_pend, tgt, lo, hi, pos_seq = [], [], [], [], [], []
    seq_char0, seq_tok0 = [], []
    nc = nt = 0
    for si, r in enumerate(records):
        out_ids = list(r["out_ids"])
        n = len(out_ids)
        st = seq_text(table, out_ids)
        u, s = codec.encode(st.text)
        _, l_, h_, t_ = position_targets(st, vocab, bool(r.get("eos_end", True)), k)
        seq_char0.append(nc); seq_tok0.append(nt)
        text_u.append(u); text_s.append(s)
        pos_c.append(st.c[:n] + nc); pos_pend.append(st.pend[:n] > 0)
        tgt.append(t_); lo.append(l_); hi.append(h_); pos_seq.append(np.full(n, si, np.int32))
        if with_cands:
            cu, cs, cl, lm, f, v = candidate_arrays(st, table, np.asarray(r["topk_ids"])[:n],
                                                    np.asarray(r["topk_logp"])[:n], u, codec,
                                                    bool(r.get("eos_end", True)))
            for k_, a in zip(("cu", "cs", "cl", "lm", "f", "v"), (cu, cs, cl, lm, f, v)):
                cand[k_].append(a)
        nc += len(u); nt += n
    pu, ps = _prompt_arrays([r.get("prompt_text", "") for r in records], codec, L_PROMPT)
    cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
    fs = FeatureSet(prompt_u=pu, prompt_s=ps, text_u=cat(text_u, np.int16), text_s=cat(text_s, np.int8),
                    seq_char0=np.array(seq_char0, np.int64), seq_tok0=np.array(seq_tok0, np.int64),
                    pos_seq=cat(pos_seq, np.int32), pos_c=cat(pos_c, np.int64), pos_pend=cat(pos_pend, bool),
                    pos_tok=np.arange(nt, dtype=np.int64),
                    tgt=np.concatenate(tgt) if tgt else np.zeros((0, k), np.int32),
                    lo=cat(lo, np.int32), hi=cat(hi, np.int32), seq_ids=[int(r["id"]) for r in records],
                    meta=dict(meta or {}, n_seq=len(records), n_pos=nt))
    if with_cands:
        fs.cand_u, fs.cand_s = np.concatenate(cand["cu"]), np.concatenate(cand["cs"])
        fs.cand_len, fs.cand_match = np.concatenate(cand["cl"]), np.concatenate(cand["lm"])
        fs.cand_f, fs.cand_valid = np.concatenate(cand["f"]), np.concatenate(cand["v"])
    return fs


def featurize_texts(texts: list[str], vocab: UnitVocab, codec: CharCodec | None = None, cuts_per_text: int = 8,
                    seed: int = 0, k: int = K_HEADS) -> FeatureSet:
    """Text-only examples from plain Hindi text: random char cuts, no prompt, no candidates, no <eos>."""
    codec = codec or CharCodec()
    rng = np.random.default_rng(seed)
    text_u, text_s, pos_c, tgt, lo, hi, pos_seq, seq_char0 = [], [], [], [], [], [], [], []
    nc = 0
    kept = 0
    for t in texts:
        if not t:
            continue
        b = t.encode("utf-8")
        _, coff = utf8_offsets(b)
        nchar = len(coff) - 1
        cuts_c = np.unique(rng.integers(0, nchar + 1, size=min(cuts_per_text, nchar + 1)))
        st = text_positions(b, coff[cuts_c])
        u, s = codec.encode(st.text)
        _, l_, h_, t_ = position_targets(st, vocab, False, k, cuts=range(len(cuts_c)))
        seq_char0.append(nc); text_u.append(u); text_s.append(s)
        pos_c.append(st.c + nc); tgt.append(t_); lo.append(l_); hi.append(h_)
        pos_seq.append(np.full(len(cuts_c), kept, np.int32))
        nc += len(u); kept += 1
    n = sum(len(x) for x in pos_c)
    return FeatureSet(prompt_u=np.zeros((kept, L_PROMPT), np.int16), prompt_s=np.zeros((kept, L_PROMPT), np.int8),
                      text_u=np.concatenate(text_u).astype(np.int16), text_s=np.concatenate(text_s).astype(np.int8),
                      seq_char0=np.array(seq_char0, np.int64), seq_tok0=np.zeros(kept, np.int64),
                      pos_seq=np.concatenate(pos_seq), pos_c=np.concatenate(pos_c).astype(np.int64),
                      pos_pend=np.zeros(n, bool), pos_tok=np.full(n, -1, np.int64),
                      tgt=np.concatenate(tgt).astype(np.int32), lo=np.concatenate(lo).astype(np.int32),
                      hi=np.concatenate(hi).astype(np.int32), seq_ids=list(range(kept)),
                      meta=dict(kind="mono", n_seq=kept, n_pos=n))


def make_batch(fs: FeatureSet, idx, use_cands: bool = True, M: int = M_LAGS, L_O: int = L_OUT) -> dict:
    """Vectorised inputs for positions ``idx`` (numpy). Shapes: ctx (B, L_P+1+L_O+1); cands (B, M*K, S)."""
    idx = np.asarray(idx)
    B = len(idx)
    seq = fs.pos_seq[idx]
    c = fs.pos_c[idx]
    offs = c[:, None] - L_O + np.arange(L_O)[None]
    ok = offs >= fs.seq_char0[seq][:, None]
    oc = np.clip(offs, 0, max(len(fs.text_u) - 1, 0))
    out_u = np.where(ok, fs.text_u[oc] if len(fs.text_u) else 0, PAD).astype(np.int64)
    out_s = np.where(ok, fs.text_s[oc] if len(fs.text_s) else 0, 0).astype(np.int64)
    pend = np.where(fs.pos_pend[idx], PEND, PAD)[:, None]
    pu = fs.prompt_u[seq].astype(np.int64)
    L_P = pu.shape[1]
    ctx_u = np.concatenate([pu, np.full((B, 1), SEP), out_u, pend], 1)
    ctx_s = np.concatenate([fs.prompt_s[seq].astype(np.int64), np.zeros((B, 1), np.int64), out_s,
                            np.zeros((B, 1), np.int64)], 1)
    ctx_t = np.concatenate([np.ones(L_P, np.int64), [2], np.full(L_O, 3), [4]])
    ctx_p = np.concatenate([np.arange(L_P)[::-1], [0], np.arange(L_O)[::-1], [0]])
    ctx_t = np.where(ctx_u == PAD, 0, ctx_t[None])
    batch = dict(ctx_u=ctx_u, ctx_s=ctx_s, ctx_t=ctx_t, ctx_p=np.broadcast_to(ctx_p, ctx_u.shape).copy(),
                 tgt=fs.tgt[idx].astype(np.int64), lo=fs.lo[idx].astype(np.int64), hi=fs.hi[idx].astype(np.int64))
    if not (use_cands and fs.has_cands):
        return batch
    K, S = fs.cand_u.shape[1], fs.cand_u.shape[2]
    tok = fs.pos_tok[idx]
    lags = np.arange(1, M + 1)
    t = tok[:, None] - lags[None]
    tv = (tok[:, None] >= 0) & (t >= fs.seq_tok0[seq][:, None])
    tc = np.clip(t, 0, None)
    cu = fs.cand_u[tc].astype(np.int64)                        # (B, M, K, S)
    cl = fs.cand_len[tc].astype(np.int64)
    d = (c[:, None] - fs.pos_c[tc])[:, :, None]                # chars committed since the anchor
    cons = fs.cand_match[tc] >= np.minimum(d, cl)
    extd = cons & (cl > d)
    sidx = d[..., None] + np.arange(S)
    ext = np.take_along_axis(cu, np.clip(sidx, 0, S - 1), axis=3)
    ext = np.where((sidx < cl[..., None]) & extd[..., None], ext, PAD)
    valid = tv[:, :, None] & fs.cand_valid[tc]
    f = np.concatenate([fs.cand_f[tc].astype(np.float32),
                        np.minimum(np.broadcast_to(d, cl.shape), 32)[..., None] / 16.0,
                        cons[..., None], extd[..., None]], -1)
    f = np.where(valid[..., None], f, 0)
    MK = M * K
    batch.update(cand_u=np.where(valid[..., None], cu, PAD).reshape(B, MK, S),
                 cand_s=fs.cand_s[tc].astype(np.int64).reshape(B, MK, S),
                 ext_u=ext.reshape(B, MK, S), cand_f=f.reshape(B, MK, -1).astype(np.float32),
                 cand_lag=np.broadcast_to(np.repeat(lags, K)[None], (B, MK)).copy(),
                 cand_rank=np.broadcast_to(np.tile(np.arange(K), M)[None], (B, MK)).copy(),
                 cand_mask=valid.reshape(B, MK))
    return batch
