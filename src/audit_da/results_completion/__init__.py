from .core import CompletionSettings, paired_panel, cluster_bootstrap, output_hash, write_outputs, sample_exclusion_manifest, _adjust_pvalues
from .architecture import estimate_accrual_architectures, build_attribution_cases, attribution_tables
from .switching import direct_revision_tables, switching_cases, switching_tables, profit_gate_sensitivity, randomisation_benchmarks, _midrank_against_reference
from .confirmatory import confirmatory_summary
from .time_shift import time_shift_benchmarks
from .applied import applied_consequence_tables, supplemental_inference

__all__ = [
    "CompletionSettings", "paired_panel", "cluster_bootstrap", "output_hash", "write_outputs", "sample_exclusion_manifest",
    "estimate_accrual_architectures", "build_attribution_cases", "attribution_tables",
    "direct_revision_tables", "switching_cases", "switching_tables", "profit_gate_sensitivity", "randomisation_benchmarks", "confirmatory_summary",
    "time_shift_benchmarks", "applied_consequence_tables", "supplemental_inference",
]
