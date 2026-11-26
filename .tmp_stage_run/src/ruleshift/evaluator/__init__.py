"""Evaluator-only package (privileged labels and references).

Week 2 provided the data adapters. Week 3 adds the shift-aware Bayes filter,
the exact finite-horizon microtask solver, a clearly labelled approximate
reference, privileged/reference baselines and the CPU phase diagram. Metrics
and closed-loop evaluation are still to come. Agent code must never import
this package.
"""

from .adapter import EvaluatorTrace, trace_from_env
from .approx_reference import ApproximateQ, TruncatedRolloutReference
from .baselines import MaxInformationGainPolicy, ReturnAwareReference, TrueMechanicOracle
from .bayes_filter import RuleShiftFilter
from .exact_solver import ExactSolver, MicrotaskConfig
from .diagnostics import DiagnosticResult, run_diagnostic
from .family_b_reference import FamilyBCosts, delta_probe_family_b
from .heldout import HeldOutResult, build_heldout_family_a, run_heldout_evaluation
from .stress import SensitivityRow, sweep_calibration, sweep_shift_prior
from .metrics_ci import PairedSummary, paired_bootstrap_ci, summarize_paired

__all__ = [
    "DiagnosticResult",
    "FamilyBCosts",
    "HeldOutResult",
    "PairedSummary",
    "ApproximateQ",
    "EvaluatorTrace",
    "ExactSolver",
    "MaxInformationGainPolicy",
    "MicrotaskConfig",
    "paired_bootstrap_ci",
    "ReturnAwareReference",
    "RuleShiftFilter",
    "SensitivityRow",
    "TruncatedRolloutReference",
    "delta_probe_family_b",
    "build_heldout_family_a",
    "run_diagnostic",
    "sweep_calibration",
    "sweep_shift_prior",
    "run_heldout_evaluation",
    "TrueMechanicOracle",
    "summarize_paired",
    "trace_from_env",
]
