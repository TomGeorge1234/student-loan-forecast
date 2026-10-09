"""Monthly projections and reproducible stochastic student-loan forecasts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import streamlit as st

from loan_rules import (
    PLAN_RULES,
    Loan,
    allocate_required_repayments,
    annual_interest_rate,
    threshold_for_tax_year,
)

TODAY_MONTH = date.today().replace(day=1)

SIMULATION_RUNS = 1_000
SIMULATION_SEED = 20_261_008
EXAMPLE_PATHS = 20
FAN_COVERAGES = np.linspace(0.05, 0.95, 31)

# Annual ONS series, 2000–2025. CHAW is the RPI All Items Index and KAB9 is
# whole-economy average weekly earnings, seasonally adjusted, total pay.
ONS_RPI_LEVELS = np.array(
    [
        170.3, 173.3, 176.2, 181.3, 186.7, 192.0, 198.1, 206.6, 214.8,
        213.7, 223.6, 235.2, 242.7, 250.1, 256.0, 258.5, 263.1, 272.5,
        281.6, 288.8, 293.1, 305.0, 340.3, 373.3, 386.7, 402.7,
    ]
)
ONS_AWE_LEVELS = np.array(
    [
        313.0, 329.0, 339.0, 350.0, 365.0, 382.0, 400.0, 420.0, 434.0,
        434.0, 444.0, 455.0, 461.0, 466.0, 471.0, 482.0, 494.0, 505.0,
        520.0, 538.0, 547.0, 580.0, 616.0, 659.0, 694.0, 728.0,
    ]
)


def fit_ar1(series: np.ndarray) -> tuple[float, float]:
    """Return OLS persistence and residual standard deviation for an AR(1)."""
    lagged = series[:-1]
    observed = series[1:]
    design = np.column_stack((np.ones(lagged.size), lagged))
    coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
    residuals = observed - design @ coefficients
    residual_std = np.sqrt(np.sum(residuals**2) / (residuals.size - 2))
    return float(coefficients[1]), float(residual_std)


HISTORICAL_RPI_GROWTH = ONS_RPI_LEVELS[1:] / ONS_RPI_LEVELS[:-1] - 1
HISTORICAL_NOMINAL_EARNINGS_GROWTH = ONS_AWE_LEVELS[1:] / ONS_AWE_LEVELS[:-1] - 1
HISTORICAL_REAL_EARNINGS_GROWTH = (
    (1 + HISTORICAL_NOMINAL_EARNINGS_GROWTH) / (1 + HISTORICAL_RPI_GROWTH) - 1
)
RPI_PERSISTENCE, RPI_INNOVATION_STD = fit_ar1(HISTORICAL_RPI_GROWTH)
REAL_SALARY_PERSISTENCE, REAL_SALARY_INNOVATION_STD = fit_ar1(
    HISTORICAL_REAL_EARNINGS_GROWTH
)

@dataclass(frozen=True)
class Row:
    month: date
    salary: float
    opening_balance: float
    interest_rate: float
    interest: float
    repayment: float
    closing_balance: float
    voluntary_payment: float = 0.0


@dataclass(frozen=True)
class Projection:
    rows: tuple[Row, ...]
    total_repaid: float
    total_interest: float
    amount_written_off: float
    outcome: str


@dataclass(frozen=True)
class SimulationSummary:
    months: tuple[date, ...]
    monthly_repayment_median: np.ndarray
    salary_median: np.ndarray
    salary_low: np.ndarray
    salary_high: np.ndarray
    salary_fan_low: np.ndarray
    salary_fan_high: np.ndarray
    balance_median: np.ndarray
    balance_low: np.ndarray
    balance_high: np.ndarray
    balance_fan_low: np.ndarray
    balance_fan_high: np.ndarray
    cumulative_median: np.ndarray
    cumulative_low: np.ndarray
    cumulative_high: np.ndarray
    cumulative_fan_low: np.ndarray
    cumulative_fan_high: np.ndarray
    salary_examples: np.ndarray
    balance_examples: np.ndarray
    cumulative_examples: np.ndarray
    total_repaid_median: float
    total_repaid_low: float
    total_repaid_high: float
    payoff_worthwhile_probability: float


def add_months(month: date, count: int = 1) -> date:
    """Return the first day of the month `count` months later."""
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def project(
    initial_salary: float,
    salary_growth: float,
    loans: tuple[Loan, ...],
    future_rpi: float,
    start_month: date = TODAY_MONTH,
    payoff_month: date | None = None,
    payoff_immediately: bool = False,
) -> Projection:
    rows: list[Row] = []
    annual_salary = initial_salary
    balances = np.array([loan.balance for loan in loans], dtype=float)[:, None]
    plans = tuple(loan.plan for loan in loans)
    total_repaid = 0.0
    total_interest = 0.0
    amount_written_off = 0.0
    month = start_month
    monthly_salary_growth = (1 + salary_growth) ** (1 / 12) - 1
    final_write_off = max(loan.write_off_month for loan in loans)
    annual_rpi = {
        year: future_rpi
        for year in range(TODAY_MONTH.year, final_write_off.year + 2)
    }

    while month <= final_write_off and np.any(balances > 0.005):
        for index, loan in enumerate(loans):
            if month >= loan.write_off_month and balances[index, 0] > 0:
                amount_written_off += balances[index, 0]
                balances[index, 0] = 0.0

        opening_balances = balances.copy()
        interest_active = np.array(
            [month < loan.write_off_month for loan in loans], dtype=bool
        )[:, None] & (balances > 0.005)
        repayment_active = interest_active & np.array(
            [month >= loan.repayment_start for loan in loans], dtype=bool
        )[:, None]
        tax_year_start = month.year if month.month >= 4 else month.year - 1
        thresholds = np.vstack(
            [
                np.atleast_1d(
                    threshold_for_tax_year(
                        PLAN_RULES[loan.plan], tax_year_start, annual_rpi
                    )
                )
                for loan in loans
            ]
        )
        salary_array = np.array([annual_salary])
        rpi_array = np.array([future_rpi])
        rates = np.vstack(
            [
                annual_interest_rate(
                    PLAN_RULES[loan.plan],
                    salary_array,
                    rpi_array,
                    thresholds[index],
                    month,
                )
                for index, loan in enumerate(loans)
            ]
        )

        is_now = payoff_immediately and month == start_month
        interest = np.where(
            interest_active & (not is_now),
            balances * ((1 + rates) ** (1 / 12) - 1),
            0.0,
        )
        balances_due = balances + interest
        repayments = (
            np.zeros_like(balances)
            if is_now
            else allocate_required_repayments(
                salary_array,
                balances_due,
                repayment_active,
                thresholds,
                plans,
            )
        )
        is_payoff_month = payoff_month is not None and month == payoff_month
        voluntary = (
            np.maximum(0.0, balances_due - repayments)
            if is_now or is_payoff_month
            else np.zeros_like(balances)
        )
        balances = np.maximum(0.0, balances_due - repayments - voluntary)
        opening_total = float(np.sum(opening_balances))
        interest_total = float(np.sum(interest))
        repayment_total = float(np.sum(repayments))
        voluntary_total = float(np.sum(voluntary))
        weighted_rate = (
            float(np.sum(rates * opening_balances) / opening_total)
            if opening_total > 0
            else 0.0
        )
        rows.append(
            Row(
                month,
                annual_salary,
                opening_total,
                weighted_rate,
                interest_total,
                repayment_total,
                float(np.sum(balances)),
                voluntary_total,
            )
        )
        total_repaid += repayment_total + voluntary_total
        total_interest += interest_total

        if not np.any(balances > 0.005):
            outcome = f"Repaid in {month.year}"
            if is_now:
                outcome = "All loans paid off now"
            elif is_payoff_month:
                outcome = f"All loans paid off in December {month.year}"
            elif amount_written_off:
                outcome = f"Final balance written off in {month.year}"
            return Projection(
                tuple(rows),
                total_repaid,
                total_interest,
                amount_written_off,
                outcome,
            )
        annual_salary *= 1 + monthly_salary_growth
        month = add_months(month)

    return Projection(
        tuple(rows),
        total_repaid,
        total_interest,
        amount_written_off + float(np.sum(balances)),
        f"Final balance written off in April {final_write_off.year}",
    )


@st.cache_data(show_spinner=False)
def simulate(
    initial_salary: float,
    mean_real_salary_growth: float,
    loans: tuple[Loan, ...],
    mean_rpi: float,
    start_month: date = TODAY_MONTH,
    payoff_month: date | None = None,
    payoff_immediately: bool = False,
    real_terms: bool = False,
    runs: int = SIMULATION_RUNS,
    totals_only: bool = False,
) -> SimulationSummary | np.ndarray:
    """Simulate growth paths, returning chart summaries or just total repayments."""
    final_write_off = max(loan.write_off_month for loan in loans)
    display_end = add_months(final_write_off, 12)
    month_count = (
        (display_end.year - start_month.year) * 12
        + display_end.month
        - start_month.month
        + 1
    )
    months = tuple(add_months(start_month, offset) for offset in range(month_count))
    # Generate inflation from the present even for a future-start forecast so that
    # thresholds whose statutory uprating begins before “Now” are indexed correctly.
    years = tuple(range(TODAY_MONTH.year, display_end.year + 1))
    year_indices = {year: index for index, year in enumerate(years)}

    rng = np.random.default_rng(SIMULATION_SEED)
    annual_rpi = np.empty((len(years), runs))
    real_salary_growth = np.empty((len(years), runs))
    previous_rpi = np.full(runs, mean_rpi)
    previous_real_growth = np.full(runs, mean_real_salary_growth)
    for year_index in range(len(years)):
        inflation_innovation = rng.normal(size=runs)
        salary_innovation = rng.normal(size=runs)
        current_rpi = (
            mean_rpi
            + RPI_PERSISTENCE * (previous_rpi - mean_rpi)
            + RPI_INNOVATION_STD * inflation_innovation
        )
        current_real_growth = (
            mean_real_salary_growth
            + REAL_SALARY_PERSISTENCE
            * (previous_real_growth - mean_real_salary_growth)
            + REAL_SALARY_INNOVATION_STD * salary_innovation
        )
        annual_rpi[year_index] = np.maximum(current_rpi, -0.99)
        real_salary_growth[year_index] = np.maximum(current_real_growth, -0.99)
        previous_rpi = annual_rpi[year_index]
        previous_real_growth = real_salary_growth[year_index]

    # Nominal pay incorporates inflation, while the two unexpected innovations
    # above remain statistically independent.
    salary_growth = np.maximum(annual_rpi + real_salary_growth, -0.99)

    annual_rpi_by_year = {
        year: annual_rpi[index] for index, year in enumerate(years)
    }

    salary = np.full(runs, initial_salary, dtype=float)
    loan_balances = np.repeat(
        np.array([loan.balance for loan in loans], dtype=float)[:, None],
        runs,
        axis=1,
    )
    plans = tuple(loan.plan for loan in loans)
    cumulative_repaid = np.zeros(runs)
    deflator = np.ones(runs)
    if not totals_only:
        salary_paths = np.empty((month_count, runs))
        balance_paths = np.empty((month_count, runs))
        cumulative_paths = np.empty((month_count, runs))
        monthly_repayment_paths = np.empty((month_count, runs))

    for offset, month in enumerate(months):
        year_index = year_indices[month.year]
        for index, loan in enumerate(loans):
            if month >= loan.write_off_month:
                loan_balances[index] = 0.0

        interest_active = np.array(
            [month < loan.write_off_month for loan in loans], dtype=bool
        )[:, None] & (loan_balances > 0.005)
        repayment_active = interest_active & np.array(
            [month >= loan.repayment_start for loan in loans], dtype=bool
        )[:, None]
        tax_year_start = month.year if month.month >= 4 else month.year - 1
        thresholds = np.vstack(
            [
                np.broadcast_to(
                    np.asarray(
                        threshold_for_tax_year(
                            PLAN_RULES[loan.plan],
                            tax_year_start,
                            annual_rpi_by_year,
                        )
                    ),
                    (runs,),
                )
                for loan in loans
            ]
        )
        rates = np.vstack(
            [
                annual_interest_rate(
                    PLAN_RULES[loan.plan],
                    salary,
                    annual_rpi[year_index],
                    thresholds[index],
                    month,
                )
                for index, loan in enumerate(loans)
            ]
        )
        is_now = payoff_immediately and month == start_month
        interest = np.where(
            interest_active & (not is_now),
            loan_balances * ((1 + rates) ** (1 / 12) - 1),
            0.0,
        )
        balances_due = loan_balances + interest
        required_repayments = (
            np.zeros_like(loan_balances)
            if is_now
            else allocate_required_repayments(
                salary,
                balances_due,
                repayment_active,
                thresholds,
                plans,
            )
        )
        is_payoff_month = payoff_month is not None and month == payoff_month
        voluntary_payments = (
            np.maximum(0.0, balances_due - required_repayments)
            if is_now or is_payoff_month
            else np.zeros_like(loan_balances)
        )
        loan_balances = np.maximum(
            0.0,
            balances_due - required_repayments - voluntary_payments,
        )
        monthly_repayment = np.sum(required_repayments + voluntary_payments, axis=0)

        display_deflator = deflator if real_terms else 1.0
        cumulative_repaid = cumulative_repaid + monthly_repayment / display_deflator
        if not totals_only:
            monthly_repayment_paths[offset] = (
                np.sum(required_repayments, axis=0) / display_deflator
            )
            salary_paths[offset] = salary / display_deflator
            balance_paths[offset] = np.sum(loan_balances, axis=0) / display_deflator
            cumulative_paths[offset] = cumulative_repaid

        salary = salary * (1 + salary_growth[year_index]) ** (1 / 12)
        deflator = deflator * (1 + annual_rpi[year_index]) ** (1 / 12)

    if totals_only:
        return cumulative_repaid

    def summarize(
        paths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lower_percentiles = (1 - FAN_COVERAGES) * 50
        upper_percentiles = (1 + FAN_COVERAGES) * 50
        quantiles = np.percentile(
            paths,
            np.concatenate(([50], lower_percentiles, upper_percentiles)),
            axis=1,
        )
        band_count = FAN_COVERAGES.size
        median = quantiles[0]
        fan_low = quantiles[1 : band_count + 1]
        fan_high = quantiles[band_count + 1 :]
        return median, fan_low[-1], fan_high[-1], fan_low, fan_high

    (
        salary_median,
        salary_low,
        salary_high,
        salary_fan_low,
        salary_fan_high,
    ) = summarize(salary_paths)
    (
        balance_median,
        balance_low,
        balance_high,
        balance_fan_low,
        balance_fan_high,
    ) = summarize(balance_paths)
    (
        cumulative_median,
        cumulative_low,
        cumulative_high,
        cumulative_fan_low,
        cumulative_fan_high,
    ) = summarize(cumulative_paths)
    final_repayments = cumulative_paths[-1]
    return SimulationSummary(
        months=months,
        monthly_repayment_median=np.median(monthly_repayment_paths, axis=1),
        salary_median=salary_median,
        salary_low=salary_low,
        salary_high=salary_high,
        salary_fan_low=salary_fan_low,
        salary_fan_high=salary_fan_high,
        balance_median=balance_median,
        balance_low=balance_low,
        balance_high=balance_high,
        balance_fan_low=balance_fan_low,
        balance_fan_high=balance_fan_high,
        cumulative_median=cumulative_median,
        cumulative_low=cumulative_low,
        cumulative_high=cumulative_high,
        cumulative_fan_low=cumulative_fan_low,
        cumulative_fan_high=cumulative_fan_high,
        salary_examples=salary_paths[:, :EXAMPLE_PATHS],
        balance_examples=balance_paths[:, :EXAMPLE_PATHS],
        cumulative_examples=cumulative_paths[:, :EXAMPLE_PATHS],
        total_repaid_median=float(np.median(final_repayments)),
        total_repaid_low=float(np.percentile(final_repayments, 2.5)),
        total_repaid_high=float(np.percentile(final_repayments, 97.5)),
        payoff_worthwhile_probability=float(
            np.mean(final_repayments > sum(loan.balance for loan in loans))
        ),
    )



@st.cache_data(show_spinner=False, max_entries=8)
def payoff_grid(
    salaries: tuple[float, ...],
    balances: tuple[float, ...],
    loans: tuple[Loan, ...],
    mean_real_salary_growth: float,
    mean_rpi: float,
    start_month: date = TODAY_MONTH,
    real_terms: bool = False,
    runs: int = SIMULATION_RUNS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare paying now with required repayments; rows are balances, columns salaries.

    Keep loan shares fixed and use identical random paths in every cell. Only
    totals are needed, avoiding the main chart's trajectory quantiles and cache.
    """
    total_balance = sum(loan.balance for loan in loans)
    if total_balance <= 0:
        raise ValueError("Enter a positive loan balance to define the loan proportions.")
    if not salaries or not balances or min(salaries) < 0 or min(balances) < 0:
        raise ValueError("Salary and balance axes must be nonempty and nonnegative.")
    probabilities = np.empty((len(balances), len(salaries)))
    savings = np.empty_like(probabilities)
    for row, balance in enumerate(balances):
        scaled_loans = tuple(
            replace(loan, balance=balance * loan.balance / total_balance)
            for loan in loans
        )
        for column, salary in enumerate(salaries):
            if balance == 0:
                totals = np.zeros(runs)
            else:
                totals = simulate.__wrapped__(
                    salary, mean_real_salary_growth, scaled_loans, mean_rpi,
                    start_month, real_terms=real_terms, runs=runs, totals_only=True,
                )
            probabilities[row, column] = np.mean(totals > balance)
            savings[row, column] = np.median(totals) - balance
    return probabilities, savings
