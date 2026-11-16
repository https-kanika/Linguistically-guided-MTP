"""A tiny deterministic byte-level tokenizer for tests (GPT-2 byte alphabet + a few merged pieces).

Pre-tokenisation: a new piece starts at every space (the space belongs to the following word), as in
Llama-3 / Qwen. Inside a piece, greedy longest match over the merged byte strings, else single bytes,
so a Devanagari char can be split across tokens (like real byte-level BPE).
"""
from lgmtp.boundaries import _bytes_to_unicode

B2U = _bytes_to_unicode()


class FakeTok:
    def __init__(self, merges=(), eos="<eos>"):
        self.vocab = [B2U[b] for b in range(256)]
        self.merges = []
        for m in merges:
            mb = m.encode("utf-8")
            self.vocab.append("".join(B2U[x] for x in mb))
            self.merges.append((mb, len(self.vocab) - 1))
        self.merges.sort(key=lambda x: -len(x[0]))
        self.vocab.append(eos)
        self.eos_token_id = len(self.vocab) - 1
        self.all_special_ids = [self.eos_token_id]
        self.added_tokens_decoder = {}
        self.pad_token = eos

    def __len__(self):
        return len(self.vocab)

    def convert_ids_to_tokens(self, ids):
        return [self.vocab[int(i)] for i in ids]

    def encode_one(self, text: str) -> list[int]:
        b = text.encode("utf-8")
        pieces, cur = [], b""
        for x in b:
            if x == 0x20 and cur:
                pieces.append(cur)
                cur = b""
            cur += bytes([x])
        if cur:
            pieces.append(cur)
        out = []
        for pc in pieces:
            i = 0
            while i < len(pc):
                for mb, mid in self.merges:
                    if pc.startswith(mb, i):
                        out.append(mid)
                        i += len(mb)
                        break
                else:
                    out.append(pc[i])
                    i += 1
        return out

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {"input_ids": self.encode_one(texts)}
        return {"input_ids": [self.encode_one(t) for t in texts]}
