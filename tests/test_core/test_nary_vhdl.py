"""
tests/test_core/test_nary_vhdl.py
====================================
N-ary generalization (core/n_ary.py) and VHDL hardware synthesis
(core/vhdl_gen.py).

Theorems verified: N-ARY-1, L4 (dim/radix regime), VHDL-93 compliance.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import unittest
from flu.core.n_ary import (
    nary_info, nary_generate, nary_generate_signed, nary_verify,
    nary_step_bound, nary_comparison_table, recommend_base,
    ternary_block_base, verify_nary_bijection,
)


# ── N-ARY-1: n-ary generalisation ────────────────────────────────────────────

def test_nary_step_bound_values():
    """N-ARY-1: step bound = min(d, floor(n/2))."""
    assert nary_step_bound(3, 2) == 1   # min(2,1)=1
    assert nary_step_bound(5, 3) == 2   # min(3,2)=2
    assert nary_step_bound(7, 5) == 3   # min(5,3)=3
    assert nary_step_bound(9, 2) == 2   # min(2,4)=2 dim-limited
    assert nary_step_bound(9, 6) == 4   # min(6,4)=4 radix-limited

def test_nary_regime_flags():
    """L4: dimension-limited vs radix-limited regime detection."""
    info_dim = nary_info(9, 2)
    assert info_dim["in_dimension_limited"] is True
    assert info_dim["in_radix_limited"] is False

    info_rad = nary_info(3, 4)
    assert info_rad["in_radix_limited"] is True

def test_nary_latin_n3_d2():
    """N-ARY-1 / T3: n=3,d=2 hyperprism is Latin."""
    result = nary_verify(3, 2)
    assert result["latin"]
    assert result["all_pass"]

def test_nary_latin_n5_d3():
    """N-ARY-1 / T3: n=5,d=3 is Latin with L1 sum=0."""
    result = nary_verify(5, 3)
    assert result["latin"]
    assert result["l1_constant_sum"]
    assert result["l1_sum_value"] == 0

def test_nary_latin_n7_d2():
    result = nary_verify(7, 2)
    assert result["all_pass"]

def test_nary_latin_n9_d2():
    result = nary_verify(9, 2)
    assert result["all_pass"], f"n=9,d=2 failed: {result}"

def test_nary_bijection_n3_d3():
    assert verify_nary_bijection(3, 3)

def test_nary_bijection_n5_d2():
    assert verify_nary_bijection(5, 2)

def test_nary_bijection_n7_d2():
    assert verify_nary_bijection(7, 2)

def test_nary_safe_limit_enforced():
    """nary_generate raises ValueError for n^d > default limit."""
    try:
        nary_generate(11, 5)   # 11^5 = 161051 > 50_000
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_nary_custom_limit_allows_larger():
    M = nary_generate(7, 3, max_cells=500)   # 343 < 500
    assert M.shape == (7, 7, 7)

def test_nary_inf_limit_bypasses():
    M = nary_generate(9, 3, max_cells=float("inf"))
    assert M.shape == (9, 9, 9)

def test_nary_signed_propagates_max_cells():
    M = nary_generate_signed(7, 3, max_cells=500)
    assert M.shape == (7, 7, 7)

def test_nary_comparison_table_structure():
    rows = nary_comparison_table([3, 5, 7], [2, 3])
    assert len(rows) == 6
    for row in rows:
        assert "step_bound" in row
        assert row["regime"] in ("dim-limited", "radix-limited")

def test_nary_recommend_base_prime():
    for n in [3, 5, 7]:
        r = recommend_base(n)
        assert r["n"] == n

def test_nary_recommend_base_prime_power():
    assert recommend_base(4)["n"] == 2    # 4=2^2 → n=2
    assert recommend_base(9)["n"] == 3    # 9=3^2 → n=3

def test_nary_ternary_block_base():
    assert ternary_block_base(1) == 3
    assert ternary_block_base(2) == 9
    assert ternary_block_base(3) == 27


# ── VHDL Generator ───────────────────────────────────────────────────────────

def _gen_vhdl(n=3, d=4):
    from flu.core.vhdl_gen import generate_vhdl
    return generate_vhdl(n=n, d=d)


def test_vhdl_entity_declaration():
    assert "entity fm_dance_core is" in _gen_vhdl()

def test_vhdl_architecture_present():
    assert "architecture rtl of fm_dance_core" in _gen_vhdl()

def test_vhdl_clock_reset_step_ports():
    vhdl = _gen_vhdl()
    for sig in ("clk", "rst", "step"):
        assert sig in vhdl, f"VHDL missing port: {sig}"

def test_vhdl_output_coords_n3_d4():
    vhdl = _gen_vhdl(3, 4)
    assert all(f"x_{i}" in vhdl for i in range(4))

def test_vhdl_n_constant():
    assert "constant N    : natural := 3" in _gen_vhdl(3, 4)

def test_vhdl_minimum_line_count():
    assert len(_gen_vhdl().splitlines()) > 50

def test_vhdl_even_n_raises():
    from flu.core.vhdl_gen import generate_vhdl
    try:
        generate_vhdl(4, 2)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_vhdl_export_creates_file():
    import tempfile, os
    from flu.core.vhdl_gen import export_vhdl
    with tempfile.NamedTemporaryFile(suffix=".vhd", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_vhdl(5, 2, tmp_path)
        assert os.path.exists(tmp_path)
        with open(tmp_path) as f:
            content = f.read()
        assert "entity fm_dance_5_2" in content
    finally:
        os.unlink(tmp_path)


# ── VHDL synthesis compliance (test class) ────────────────────────────────────

class TestVHDLSynthesisFixes(unittest.TestCase):

    def _vhdl(self, n=3, d=4):
        from flu.core.vhdl_gen import generate_vhdl
        return generate_vhdl(n=n, d=d)

    def test_no_declare_block_inside_elsif(self):
        """VHDL-93: no 'declare' block inside elsif."""
        vhdl = self._vhdl()
        lines = vhdl.splitlines()
        for i, line in enumerate(lines):
            if "elsif" in line.lower() and "step" in line.lower():
                window = "\n".join(lines[i:i+6])
                self.assertNotIn("declare", window.lower())

    def test_no_mod_n_in_arithmetic(self):
        """No 'mod N' hardware divider in non-comment RTL."""
        vhdl = self._vhdl()
        non_comment = "\n".join(l for l in vhdl.splitlines() if not l.strip().startswith("--"))
        self.assertNotIn("mod N", non_comment)

    def test_carry_variable_in_declarative_region(self):
        """'carry' variable in process declarative region (before begin)."""
        vhdl = self._vhdl()
        idx_proc  = vhdl.find("counter_proc : process")
        idx_begin = vhdl.find("begin", idx_proc)
        decl = vhdl[idx_proc:idx_begin]
        self.assertIn("carry", decl)

    def test_if_accum_ge_n_present(self):
        """Modulo-free wrap uses 'if accum >= N'."""
        self.assertIn("accum >= N", self._vhdl())

    def test_required_structural_elements(self):
        """Structural regression: core elements present after fixes."""
        vhdl = self._vhdl(n=5, d=3)
        for elem in ("entity fm_dance_core is", "architecture rtl",
                     "counter_proc", "transform_proc", "rising_edge"):
            self.assertIn(elem, vhdl, f"Missing: {elem}")


# ── Package import smoke test ─────────────────────────────────────────────────

class TestImportSmoke(unittest.TestCase):

    def test_py_typed_marker_exists(self):
        """PEP 561 py.typed marker present."""
        import pathlib
        marker = pathlib.Path(__file__).parent.parent.parent / "src" / "flu" / "py.typed"
        self.assertTrue(marker.exists(), f"py.typed missing at {marker}")

    def test_flu_imports_cleanly(self):
        """Top-level import exposes all expected symbols."""
        import flu
        for sym in ("generate", "traverse", "FLUHyperCell", "SparseCommunionManifold",
                    "ScarStore", "FractalNet", "DynamicFLUNetwork"):
            self.assertTrue(hasattr(flu, sym), f"flu.{sym} missing")

    def test_interfaces_import_cleanly(self):
        """All V15 interface modules import without error."""
        for mod in ("lexicon", "integrity", "genetic", "invariance",
                    "hilbert", "cohomology", "gray_code", "crypto", "hadamard"):
            importlib = __import__("importlib")
            m = importlib.import_module(f"flu.interfaces.{mod}")
            self.assertIsNotNone(m)
