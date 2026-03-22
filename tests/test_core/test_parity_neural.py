"""
tests/test_core/test_parity_neural.py
=======================================
Parity Switcher (core/parity_switcher.py) and Dynamic Neural Scaling
(applications/neural.py).  Tests were previously embedded in the manual
bedrock runner; now collected as proper test_ functions.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np


# ── Parity Switcher ──────────────────────────────────────────────────────────

def test_ps_odd_n_mean_zero():
    """T-PS-01: odd n dispatches to FM-Dance; signed mean ≈ 0."""
    from flu.core.parity_switcher import generate
    for n, d in [(3, 2), (5, 3), (7, 2)]:
        M = generate(n, d, signed=True)
        assert abs(float(np.mean(M))) < 1e-9, f"PS odd mean non-zero n={n},d={d}"

def test_ps_even_n_correct_shape():
    """T-PS-02: even n dispatches to sum-mod; shape = (n,)*d."""
    from flu.core.parity_switcher import generate
    for n, d in [(4, 2), (6, 2), (8, 3)]:
        M = generate(n, d, signed=False)
        assert M.shape == (n,)*d, f"PS shape mismatch n={n},d={d}: {M.shape}"

def test_ps_latin_property():
    """T-PS-03: verify_latin passes for both odd and even n."""
    from flu.core.parity_switcher import verify_latin
    for n, d in [(3, 2), (5, 2), (4, 2), (6, 2), (7, 3)]:
        r = verify_latin(n, d)
        assert r["latin_ok"], f"PS verify_latin failed n={n},d={d}"

def test_ps_metadata_branch():
    """T-PS-04: metadata branch field correct for odd/even n."""
    from flu.core.parity_switcher import generate_metadata
    for n, expected in [(3, "fm_dance"), (5, "fm_dance"), (4, "even_kronecker"), (6, "even_kronecker")]:
        meta = generate_metadata(n, 2)
        assert meta["branch"] == expected, f"PS branch n={n}: {meta['branch']}"

def test_ps_metadata_hamiltonian_flag():
    """T-PS-05: hamiltonian=True for odd n, False for even n."""
    from flu.core.parity_switcher import generate_metadata
    assert generate_metadata(3, 2)["hamiltonian"] is True
    assert generate_metadata(4, 2)["hamiltonian"] is False

def test_ps_metadata_step_bound():
    """T-PS-06: step_bound is correct for odd n; None for even n."""
    from flu.core.parity_switcher import generate_metadata
    meta_odd  = generate_metadata(5, 3)
    meta_even = generate_metadata(4, 3)
    assert meta_odd["step_bound"] == min(3, 5 // 2)
    assert meta_even["step_bound"] is None

def test_ps_raises_on_n_lt_2():
    """T-PS-07: generate raises ValueError for n < 2."""
    from flu.core.parity_switcher import generate
    try:
        generate(1, 2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_ps_accessible_from_flu_toplevel():
    """T-PS-08: flu.generate() is exported."""
    import flu
    M = flu.generate(5, 2)
    assert M.shape == (5, 5)

def test_ps_verify_latin_signed():
    """T-PS-09: verify_latin with signed=True passes for both parities."""
    from flu.core.parity_switcher import verify_latin
    for n in [3, 4, 5, 6, 7]:
        r = verify_latin(n, 2, signed=True)
        assert r["latin_ok"], f"PS verify_latin signed n={n}"


# ── Dynamic Neural Scaling ───────────────────────────────────────────────────

def test_nn_basic_layer_creation():
    """T-NN-01: add_layer and layer_weights basic smoke test."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=7, in_features=7)
    net.add_layer("fc1", out_features=7)
    W1 = net.layer_weights("fc1")
    assert W1.shape == (7, 7)

def test_nn_expansion_preserves_existing_layers():
    """T-NN-02: adding new layers does not alter existing layer weights."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=5, in_features=5)
    net.add_layer("fc1")
    W1_before = net.layer_weights("fc1").copy()
    net.add_layer("fc2")
    net.add_layer("fc3")
    W1_after  = net.layer_weights("fc1")
    assert np.allclose(W1_before, W1_after)

def test_nn_different_layers_have_different_seeds():
    """T-NN-03: different layers get different weight matrices."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=7, in_features=7)
    net.add_layer("a").add_layer("b").add_layer("c")
    Wa, Wb, Wc = net.layer_weights("a"), net.layer_weights("b"), net.layer_weights("c")
    assert not np.allclose(Wa, Wb), "Layers a and b should differ"
    assert not np.allclose(Wa, Wc), "Layers a and c should differ"

def test_nn_add_layer_chaining():
    """T-NN-04: add_layer returns self (method chaining)."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=3, in_features=3)
    assert net.add_layer("x") is net

def test_nn_duplicate_name_raises():
    """T-NN-05: adding a duplicate layer name raises ValueError."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=3, in_features=3)
    net.add_layer("same")
    try:
        net.add_layer("same")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_nn_unknown_layer_raises():
    """T-NN-06: layer_weights for unknown layer raises KeyError."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=3, in_features=3)
    try:
        net.layer_weights("ghost")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

def test_nn_all_weights_keys():
    """T-NN-07: all_weights returns dict keyed by layer name."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=5, in_features=5)
    for name in ["l1", "l2", "l3"]:
        net.add_layer(name)
    W = net.all_weights()
    assert set(W.keys()) == {"l1", "l2", "l3"}
    assert all(v.shape == (5, 5) for v in W.values())

def test_nn_expansion_report_structure():
    """T-NN-08: expansion_report has correct keys and values."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=5, in_features=5)
    net.add_layer("x").add_layer("y")
    report = net.expansion_report()
    assert report["layers_registered"] == 2
    assert "PROVEN" in report["latin_property"]
    assert report["seed_reservoir_capacity"] == 120

def test_nn_golden_seed_strategy():
    """T-NN-09: golden seed strategy produces correct shape."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=5, in_features=5, seed_strategy="golden")
    net.add_layer("g1").add_layer("g2")
    assert net.layer_weights("g1").shape == (5, 5)

def test_nn_flu_initializer_parity_switcher():
    """T-NN-10: FLUInitializer uses parity_switcher; correct shape + variance."""
    from flu.applications.neural import FLUInitializer
    init = FLUInitializer(signed=True)
    W = init.weights((5, 5))
    assert W.shape == (5, 5)
    assert abs(float(W.std()) - 1.0) < 0.1

def test_nn_exported_from_flu_toplevel():
    """T-NN-11: flu.DynamicFLUNetwork is exported at top level."""
    import flu
    net2 = flu.DynamicFLUNetwork(n=3, in_features=3)
    net2.add_layer("test")
    assert net2.layer_weights("test").shape == (3, 3)

def test_nn_even_n_network():
    """T-NN-12: even n networks work; layers get distinct weights."""
    from flu.applications.neural import DynamicFLUNetwork
    net = DynamicFLUNetwork(n=4, in_features=4)
    net.add_layer("e1").add_layer("e2")
    We1, We2 = net.layer_weights("e1"), net.layer_weights("e2")
    assert We1.shape == (4, 4)
    assert not np.allclose(We1, We2)
