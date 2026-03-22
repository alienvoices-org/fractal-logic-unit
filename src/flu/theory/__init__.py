from flu.theory.theory import PhasedFractalNumberTheory
from flu.theory.theory_fm_dance import (
    ModularLattice,
    TheoremRecord,
    ALL_THEOREMS,
    ALL_CONJECTURES,
    HISTORICAL_MAPPING,
    MATHEMATICAL_CONNECTIONS,
    T1_BIJECTION, T2_HAMILTONIAN, T3_LATIN_HYPERCUBE,
    T4_STEP_BOUND, T5_SIAMESE, T6_FRACTAL,
    C1_HOLOGRAPHIC_REPAIR, C2_SPECTRAL_UNIFORMITY,
    C3_TENSOR_CLOSURE, C4_TORUS_CYCLE,
    theorem_status_report,
)
from flu.theory.theorem_registry import (
    REGISTRY,
    get_theorem,
    proven_theorems,
    open_conjectures,
    status_report,
)
from flu.theory.theory_latin import (
    verify_holographic_repair,
    verify_constant_line_sum,
    verify_all_latin_theorems,
    holographic_repair,
    line_sum_constant,
    byzantine_fault_tolerance_degree,
)
from flu.theory.theory_spectral import (
    verify_spectral_flatness,
    verify_dc_zero,
    verify_all_spectral,
    compute_spectral_profile,
)
from flu.theory.theory_container import (
    PermutationLattice,
    GENERATOR_ROLES,
    PERMUTATION_LATTICE_ALGEBRA,
    permutation_lattice_summary,
)
