"""UK student-loan plan rules and multi-plan repayment allocation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Mapping

import numpy as np


PLAN_1 = "Plan 1"
PLAN_2 = "Plan 2"
PLAN_4 = "Plan 4"
PLAN_5 = "Plan 5"
POSTGRADUATE = "Postgraduate"
PLAN_NAMES = (PLAN_1, PLAN_2, PLAN_4, PLAN_5, POSTGRADUATE)

CURRENT_TAX_YEAR = 2026
CURRENT_RPI = 0.041
CURRENT_RATE_END = date(2027, 9, 1)
PLAN_2_UPPER_THRESHOLD = 52_885.0


@dataclass(frozen=True)
class PlanRule:
    name: str
    current_threshold: float
    repayment_rate: float
    repayment_group: str
    write_off_years: int
    threshold_anchor_year: int
    threshold_anchor_value: float
    threshold_index_start: int | None
    interest_kind: str


PLAN_RULES: Mapping[str, PlanRule] = {
    PLAN_1: PlanRule(
        PLAN_1, 26_900.0, 0.09, "undergraduate", 25, 2027, 28_005.0, 2028, "rpi_capped"
    ),
    PLAN_2: PlanRule(
        PLAN_2, 29_385.0, 0.09, "undergraduate", 30, 2029, 29_385.0, 2030, "plan_2"
    ),
    PLAN_4: PlanRule(
        PLAN_4, 33_795.0, 0.09, "undergraduate", 30, 2026, 33_795.0, 2027, "rpi_capped"
    ),
    PLAN_5: PlanRule(
        PLAN_5, 25_000.0, 0.09, "undergraduate", 40, 2026, 25_000.0, 2027, "rpi"
    ),
    POSTGRADUATE: PlanRule(
        POSTGRADUATE, 21_000.0, 0.06, "postgraduate", 30, 2026, 21_000.0, None, "rpi_plus_3"
    ),
}


@dataclass(frozen=True)
class Loan:
    plan: str
    balance: float
    graduation_date: date

    @property
    def rule(self) -> PlanRule:
        return PLAN_RULES[self.plan]

    @property
    def repayment_start(self) -> date:
        april = date(self.graduation_date.year, 4, 1)
        if april <= self.graduation_date:
            april = date(self.graduation_date.year + 1, 4, 1)
        if self.plan == PLAN_5:
            april = max(april, date(2026, 4, 1))
        return april

    @property
    def write_off_month(self) -> date:
        return date(
            self.repayment_start.year + self.rule.write_off_years,
            4,
            1,
        )


def threshold_for_tax_year(
    rule: PlanRule,
    tax_year: int,
    annual_rpi: Mapping[int, float | np.ndarray],
) -> float | np.ndarray:
    """Return the assumed UK repayment threshold for a tax year."""
    if tax_year <= CURRENT_TAX_YEAR:
        return rule.current_threshold
    if tax_year <= rule.threshold_anchor_year:
        return rule.threshold_anchor_value
    if rule.threshold_index_start is None:
        return rule.threshold_anchor_value

    threshold: float | np.ndarray = rule.threshold_anchor_value
    for year in range(rule.threshold_index_start, tax_year + 1):
        threshold = threshold * (1 + annual_rpi[year])
    return threshold


def annual_interest_rate(
    rule: PlanRule,
    salary: np.ndarray,
    rpi: np.ndarray,
    threshold: np.ndarray,
    month: date,
) -> np.ndarray:
    """Return annual interest rates for one plan across simulation paths."""
    if month < CURRENT_RATE_END:
        if rule.name == PLAN_2:
            upper = threshold * (PLAN_2_UPPER_THRESHOLD / PLAN_RULES[PLAN_2].current_threshold)
            fraction = np.clip((salary - threshold) / (upper - threshold), 0.0, 1.0)
            return CURRENT_RPI + (0.06 - CURRENT_RPI) * fraction
        if rule.name == POSTGRADUATE:
            return np.full_like(salary, 0.06)
        return np.full_like(salary, CURRENT_RPI)

    if rule.interest_kind == "plan_2":
        upper = threshold * (PLAN_2_UPPER_THRESHOLD / PLAN_RULES[PLAN_2].current_threshold)
        extra = np.clip(0.03 * (salary - threshold) / (upper - threshold), 0.0, 0.03)
        return rpi + extra
    if rule.interest_kind == "rpi_plus_3":
        return rpi + 0.03
    # Plans 1 and 4 are technically the lower of RPI and Bank Rate + 1%.
    # In the absence of a Bank Rate model, the forecast uses the RPI branch.
    return rpi


def allocate_required_repayments(
    salary: np.ndarray,
    balances_due: np.ndarray,
    active: np.ndarray,
    thresholds: np.ndarray,
    plans: tuple[str, ...],
) -> np.ndarray:
    """Allocate PAYE deductions across active loan plans.

    Undergraduate plans share one 9% deduction above the lowest active threshold.
    Threshold-band caps allocate that deduction from the lowest-threshold plan upward.
    A postgraduate loan receives a separate 6% deduction.
    """
    payments = np.zeros_like(balances_due)
    undergraduate = [
        index
        for index, plan in enumerate(plans)
        if PLAN_RULES[plan].repayment_group == "undergraduate"
    ]
    undergraduate.sort(key=lambda index: PLAN_RULES[plans[index]].current_threshold)

    if undergraduate:
        active_thresholds = np.where(active[undergraduate], thresholds[undergraduate], np.inf)
        lowest = np.min(active_thresholds, axis=0)
        remaining = np.where(
            np.isfinite(lowest),
            np.maximum(0.0, salary - lowest) * 0.09 / 12,
            0.0,
        )

        for position, index in enumerate(undergraduate):
            is_active = active[index]
            if position + 1 < len(undergraduate):
                later = undergraduate[position + 1 :]
                next_threshold = np.min(
                    np.where(active[later], thresholds[later], np.inf),
                    axis=0,
                )
                band_cap = np.where(
                    np.isfinite(next_threshold),
                    np.maximum(0.0, next_threshold - thresholds[index]) * 0.09 / 12,
                    remaining,
                )
            else:
                band_cap = remaining
            payment = np.where(
                is_active,
                np.minimum(balances_due[index], np.minimum(remaining, band_cap)),
                0.0,
            )
            payments[index] = payment
            remaining = np.maximum(0.0, remaining - payment)

    for index, plan in enumerate(plans):
        if PLAN_RULES[plan].repayment_group != "postgraduate":
            continue
        deduction = np.maximum(0.0, salary - thresholds[index]) * 0.06 / 12
        payments[index] = np.where(
            active[index],
            np.minimum(balances_due[index], deduction),
            0.0,
        )

    return payments
