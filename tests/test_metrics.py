import numpy as np
import pytest

from jspace_quant import metrics, readout


def test_kl_zero_for_identical_and_positive_otherwise():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    lp = readout.log_softmax(logits)
    np.testing.assert_allclose(metrics.kl_divergence(lp, lp), 0.0, atol=1e-12)
    other = readout.log_softmax(logits[:, ::-1])
    assert (metrics.kl_divergence(lp, other) > 0).all()


def test_log_softmax_normalized_and_stable():
    lp = readout.log_softmax(np.array([[1000.0, 1001.0, 999.0]]))
    np.testing.assert_allclose(np.exp(lp).sum(), 1.0)


def test_topk_ids_sorted_descending():
    logits = np.array([[0.1, 0.9, 0.5, 0.7, 0.3]])
    assert readout.topk_ids(logits, 3).tolist() == [[1, 3, 2]]


def test_min_rank_takes_best_candidate_and_is_one_based():
    logits = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    assert readout.min_rank(logits, [0]).tolist() == [1]
    assert readout.min_rank(logits, [3, 2]).tolist() == [3]  # best candidate counts
    assert readout.min_rank(np.array([[1.0, 1.0, 0.0]]), [1]).tolist() == [1]  # tie -> candidate


def test_topk_overlap():
    a = np.array([[1, 2, 3, 4]])
    b = np.array([[4, 3, 9, 8]])
    assert metrics.topk_overlap(a, b).tolist() == [0.5]


def test_pass_at_k_is_mean_of_per_item_fractions():
    ranks = [np.array([1, 50]), np.array([3])]
    assert metrics.pass_at_k(ranks, 1) == pytest.approx((0.5 + 0.0) / 2)
    assert metrics.pass_at_k(ranks, 10) == pytest.approx((0.5 + 1.0) / 2)


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    v = np.random.default_rng(1).normal(size=200)
    lo, hi = metrics.bootstrap_ci(v, n=500, seed=0)
    assert lo < v.mean() < hi
    assert (lo, hi) == metrics.bootstrap_ci(v, n=500, seed=0)


def test_cluster_bootstrap_is_wider_for_correlated_clusters():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(20), 10)
    v = rng.normal(size=20)[clusters] + 0.01 * rng.normal(size=200)  # clusters nearly constant
    lo_i, hi_i = metrics.bootstrap_ci(v, n=1000, seed=0)
    lo_c, hi_c = metrics.cluster_bootstrap_ci(v, clusters, n=1000, seed=0)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_workspace_band_contiguous_around_peak_excluding_last():
    rate = np.array([0.0, 0.1, 0.4, 0.8, 0.5, 0.2, 0.6, 0.9])  # last = output, ignored
    assert metrics.workspace_band(rate) == (2, 4)


def test_floor_sets_flags_sets_where_reference_fails():
    import pandas as pd

    from jspace_quant import analysis, config

    ref = config.REFERENCE_ARM
    rows = []
    for item, (s, rank) in enumerate([("good", 3), ("good", 50), ("bad", 500), ("bad", 900)]):
        rows += [{"arm": ref, "set": s, "item": item, "word": "w", "layer": l, "rank": rank} for l in range(3)]
        rows += [{"arm": "other", "set": s, "item": item, "word": "w", "layer": 0, "rank": 1}]
    assert analysis.floor_sets(pd.DataFrame(rows)) == ["bad"]


def test_linear_cka_identity_rotation_and_unrelated():
    from jspace_quant.metrics import linear_cka

    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 20))
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, 3.0 * x @ q) == pytest.approx(1.0)  # invariant to rotation and scaling
    assert linear_cka(x, rng.normal(size=(300, 20))) < 0.2
