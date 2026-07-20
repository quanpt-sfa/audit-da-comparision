from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class AnalysisWindow:
    """Single source of truth for the TT200 empirical time contract.

    Source and target construction may use 2015-2025. Rolling models are trained
    only on observations from 2015 onward and are evaluated out of sample from
    2016 onward. The effective first test year may be later when a prespecified
    minimum training-sample gate is not met.
    """

    source_start_year: int = 2015
    source_end_year: int = 2025
    training_start_year: int = 2015
    test_start_year: int = 2016
    test_end_year: int = 2025

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        fallback: Mapping[str, Any] | None = None,
    ) -> "AnalysisWindow":
        values: dict[str, Any] = {}
        if fallback:
            values.update(fallback)
        if mapping:
            values.update(mapping)
        window = cls(
            source_start_year=int(values.get("source_start_year", 2015)),
            source_end_year=int(values.get("source_end_year", 2025)),
            training_start_year=int(values.get("training_start_year", 2015)),
            test_start_year=int(values.get("test_start_year", 2016)),
            test_end_year=int(values.get("test_end_year", 2025)),
        )
        window.validate()
        return window

    def validate(self) -> None:
        if self.source_start_year > self.source_end_year:
            raise ValueError("source_start_year must not exceed source_end_year")
        if self.training_start_year < self.source_start_year:
            raise ValueError(
                "training_start_year cannot precede source_start_year under the "
                "TT200-only estimation design"
            )
        if self.training_start_year >= self.test_start_year:
            raise ValueError("training_start_year must precede test_start_year")
        if self.test_start_year > self.test_end_year:
            raise ValueError("test_start_year must not exceed test_end_year")
        if self.test_end_year > self.source_end_year:
            raise ValueError("test_end_year cannot exceed source_end_year")

    def source_mask(self, years: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(years, errors="coerce")
        return numeric.between(
            self.source_start_year, self.source_end_year, inclusive="both"
        )

    def target_mask(self, years: pd.Series) -> pd.Series:
        return self.source_mask(years)

    def training_mask(self, years: pd.Series, test_year: int) -> pd.Series:
        numeric = pd.to_numeric(years, errors="coerce")
        return numeric.between(
            self.training_start_year, int(test_year) - 1, inclusive="both"
        )

    def test_mask(self, years: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(years, errors="coerce")
        return numeric.between(
            self.test_start_year, self.test_end_year, inclusive="both"
        )

    def test_years(self) -> range:
        return range(self.test_start_year, self.test_end_year + 1)

    def as_dict(self) -> dict[str, int]:
        return {
            "source_start_year": self.source_start_year,
            "source_end_year": self.source_end_year,
            "training_start_year": self.training_start_year,
            "test_start_year": self.test_start_year,
            "test_end_year": self.test_end_year,
        }


def window_from_section(
    section: Mapping[str, Any],
    *,
    legacy_source_start: str = "minimum_year",
    legacy_source_end: str = "maximum_year",
    legacy_test_start: str = "minimum_test_year",
    legacy_test_end: str = "maximum_test_year",
) -> AnalysisWindow:
    nested = section.get("analysis_window", {})
    fallback = {
        "source_start_year": section.get(legacy_source_start, 2015),
        "source_end_year": section.get(legacy_source_end, 2025),
        "training_start_year": section.get("training_start_year", 2015),
        "test_start_year": section.get(legacy_test_start, 2016),
        "test_end_year": section.get(legacy_test_end, 2025),
    }
    return AnalysisWindow.from_mapping(nested, fallback=fallback)
