import numpy as np
from jlens_gguf.readout import LensReadout

from jspace_quant import intervene


class _FakeReadout:
    """Minimal stand-in exposing lens_vector, enough for LensReadout's factor methods."""

    def __init__(self, vectors):
        self.vectors = vectors

    def lens_vector(self, layer, token_id, *, unit=True):
        v = self.vectors[token_id]
        return v / np.linalg.norm(v) if unit else v


def test_swap_factors_exchange_the_two_lens_coordinates():
    rng = np.random.default_rng(0)
    vecs = {1: rng.normal(size=16), 2: rng.normal(size=16)}
    fake = _FakeReadout(vecs)
    A, B = LensReadout.swap_factors(fake, 0, 1, 2)
    V = np.stack([vecs[1], vecs[2]], axis=1)
    h = V @ np.array([3.0, -1.0]) + 0.1 * rng.normal(size=16)
    h_new = h + A @ (B @ h)
    c, c_new = np.linalg.pinv(V) @ h, np.linalg.pinv(V) @ h_new
    np.testing.assert_allclose(c_new, c[::-1], atol=1e-5)
    # the part of h outside span(V) is untouched
    P = V @ np.linalg.pinv(V)
    np.testing.assert_allclose(h_new - P @ h_new, h - P @ h, atol=1e-5)


def test_ablate_factors_remove_the_direction():
    rng = np.random.default_rng(1)
    v = rng.normal(size=16)
    A, B = LensReadout.ablate_factors(_FakeReadout({5: v}), 0, [5])
    h = rng.normal(size=16)
    h_new = h + A @ (B @ h)
    assert abs(h_new @ v) < 1e-5


def test_concept_token_prefers_single_token_with_space():
    vocab = {" Brazil": [7], "Brazil": [8], " Portuguese": [1, 2], "Portuguese": [3, 4]}
    tok = lambda s: vocab.get(s, [0, 0])  # noqa: E731
    assert intervene.concept_token("Brazil", tok) == 7
    assert intervene.concept_token("Portuguese", tok) is None


def test_control_concept_is_unrelated():
    items = [{"intermediate": "Brazil", "swap_to": "Mexico"}, {"intermediate": "Mexico", "swap_to": "Brazil"},
             {"intermediate": "turtle", "swap_to": "snake"}]
    assert intervene.control_concept(items, 0) == "snake"
