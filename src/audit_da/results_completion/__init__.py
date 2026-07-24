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
from .method_locked import (
    build_attribution_cases,
    estimate_accrual_architectures,
    randomisation_benchmarks,
)
from .method_contract import (
    LOCKED_METHOD_CONTRACT,
    method_contract_sha256,
    validate_method_contract,
)
from .switching import (
    _midrank_against_reference,
    direct_revision_tables,
)
from .switching_complete_case import (
    profit_gate_sensitivity,
    switching_cases,
)
from .parallel import attribution_tables, switching_tables
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
    "LOCKED_METHOD_CONTRACT",
    "validate_method_contract",
    "method_contract_sha256",
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
