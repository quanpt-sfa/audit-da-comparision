from .core import (
    CompletionSettings,
    _adjust_pvalues,
    cluster_bootstrap,
    cluster_bootstrap_1d,
    output_hash,
    paired_panel,
    sample_exclusion_manifest,
    write_outputs,
)
from .architecture import estimate_accrual_architectures, build_attribution_cases
from .switching import (
    _midrank_against_reference,
    direct_revision_tables,
    profit_gate_sensitivity,
    switching_cases,
)
from .parallel import attribution_tables, randomisation_benchmarks, switching_tables
from .confirmatory import confirmatory_summary
from .time_shift import time_shift_benchmarks
from .applied import applied_consequence_tables, supplemental_inference

__all__ = [
    "CompletionSettings",
    "paired_panel",
    "cluster_bootstrap",
    "cluster_bootstrap_1d",
    "output_hash",
    "write_outputs",
    "sample_exclusion_manifest",
    "estimate_accrual_architectures",
    "build_attribution_cases",
    "attribution_tables",
    "direct_revision_tables",
    "switching_cases",
    "switching_tables",
    "profit_gate_sensitivity",
    "randomisation_benchmarks",
    "confirmatory_summary",
    "time_shift_benchmarks",
    "applied_consequence_tables",
    "supplemental_inference",
]
