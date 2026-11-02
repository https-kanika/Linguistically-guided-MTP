import pytest

torch = pytest.importorskip("torch")
from transformers import DynamicCache  # noqa: E402

from lgmtp.spec_loop import LiveView, crop_to, first_divergence  # noqa: E402
from lgmtp.simulator import CausalityError  # noqa: E402


def _cache(n, layers=2):
    c = DynamicCache()
    for li in range(layers):
        k = torch.arange(n, dtype=torch.float32).view(1, 1, n, 1).repeat(1, 2, 1, 4)
        c.update(k, k.clone(), li)
    return c


def test_crop_to_removes_tail_tokens():
    c = _cache(10)
    crop_to(c, 7)
    assert c.get_seq_length() == 7
    k = c.layers[1].keys
    assert k.shape[-2] == 7 and k[0, 0, :, 0].tolist() == list(range(7))   # kept the *first* 7 positions
    crop_to(c, 9)                       # longer than current: no-op
    assert c.get_seq_length() == 7
    crop_to(c, 7)                       # equal: no-op
    assert c.get_seq_length() == 7


def test_live_view_causality():
    v = LiveView()
    v.out, v.p, v.limit = [5, 6], 2, 3
    v.draft = {2: 7, 3: 8}
    v.feats = {2: {"H": 0.1}, 3: {"H": 0.2}}
    assert v.token(1) == 6 and v.token(3) == 8 and v.feat("H", 3) == 0.2
    with pytest.raises(CausalityError):
        v.token(4)
    with pytest.raises(CausalityError):
        v.feat("H", 1)                  # committed by the target: not computed live
    with pytest.raises(CausalityError):
        v.gold("match", 2)


def test_first_divergence():
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2, 3], [1, 9, 3]) == 1
    assert first_divergence([1, 2], [1, 2, 3]) == 2
