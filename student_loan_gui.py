"""Interactive multi-plan UK student-loan forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import altair as alt
import numpy as np
import streamlit as st

from loan_rules import (
    PLAN_1,
    PLAN_2,
    PLAN_4,
    PLAN_5,
    PLAN_NAMES,
    PLAN_RULES,
    POSTGRADUATE,
    Loan,
    allocate_required_repayments,
    annual_interest_rate,
    threshold_for_tax_year,
)

TODAY_MONTH = date.today().replace(day=1)

# Fixed samples from Seaborn's perceptually uniform crest and flare colour maps.
SALARY_COLOR = "#3C6682"  # crest
LOAN_COLOR = "#C14168"  # flare
REPAID_COLOR = "#C8872D"

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
) -> SimulationSummary:
    """Simulate mean-reverting inflation and real-salary growth paths."""
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
        monthly_repayment_paths[offset] = (
            np.sum(required_repayments, axis=0) / display_deflator
        )
        cumulative_repaid = cumulative_repaid + monthly_repayment / display_deflator
        salary_paths[offset] = salary / display_deflator
        balance_paths[offset] = np.sum(loan_balances, axis=0) / display_deflator
        cumulative_paths[offset] = cumulative_repaid

        salary = salary * (1 + salary_growth[year_index]) ** (1 / 12)
        deflator = deflator * (1 + annual_rpi[year_index]) ** (1 / 12)

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


def money(value: float) -> str:
    return f"£{round(value / 100) * 100:,.0f}"


def signed_money(value: float) -> str:
    rounded = round(value / 100) * 100
    if rounded > 0:
        return f"+{money(rounded)}"
    if rounded < 0:
        return f"−{money(abs(rounded))}"
    return money(0)


def money_range_html(
    central: float,
    low: float,
    high: float,
    signed: bool = False,
) -> str:
    """Format a central value with its absolute lower and upper estimates."""
    formatter = signed_money if signed else money
    return (
        f"{formatter(central)}"
        f'<span class="range-errors">'
        f"<sup>high: {formatter(high)}</sup>"
        f"<sub>low: {formatter(low)}</sub>"
        f"</span>"
    )


def payoff_verdict(probability: float, payoff_immediately: bool = False) -> str:
    if payoff_immediately:
        return "Selected"
    if probability >= 0.95:
        return "Yes"
    if probability >= 2 / 3:
        return "Probably"
    if probability >= 1 / 3:
        return "It's unclear"
    if probability > 0.05:
        return "Probably not"
    return "No"


@st.cache_data(show_spinner=False)
def make_simulation_chart(
    summary: SimulationSummary,
    loan_write_offs: tuple[tuple[str, date], ...],
    real_terms: bool = False,
) -> dict:
    series = (
        (
            summary.salary_median,
            summary.salary_fan_low,
            summary.salary_fan_high,
            SALARY_COLOR,
            "Salary",
        ),
        (
            summary.balance_median,
            summary.balance_fan_low,
            summary.balance_fan_high,
            LOAN_COLOR,
            "Loan balance",
        ),
        (
            summary.cumulative_median,
            summary.cumulative_fan_low,
            summary.cumulative_fan_high,
            REPAID_COLOR,
            "Repaid (cumulative)",
        ),
    )
    band_values = [
        {
            "month": month.isoformat(),
            "low": float(low[index]),
            "high": float(high[index]),
            "series": label,
            "band": band_index,
        }
        for _median, fan_low, fan_high, _color, label in series
        for band_index, (low, high) in enumerate(zip(fan_low, fan_high))
        for index, month in enumerate(summary.months)
    ]
    color_scale = alt.Scale(
        domain=[item[4] for item in series],
        range=[item[3] for item in series],
    )
    band_chart = (
        alt.Chart(alt.Data(values=band_values))
        .mark_area(opacity=0.018)
        .encode(
            x=alt.X("month:T", title=None),
            y=alt.Y(
                "low:Q",
                title=None,
                scale=alt.Scale(zero=True),
                axis=alt.Axis(labelExpr="'£' + format(datum.value, '~s')"),
            ),
            y2="high:Q",
            color=alt.Color(
                "series:N",
                scale=color_scale,
                legend=None,
            ),
            detail="band:N",
        )
    )

    legend_layers = [
        alt.Chart(alt.Data(values=[{}]))
        .mark_rect(color="#dddddd", opacity=0.78, cornerRadius=3)
        .encode(
            x=alt.value(35),
            x2=alt.value(395),
            y=alt.value(-50),
            y2=alt.value(-24),
        )
    ]
    legend_positions = (
        (55, "Salary", SALARY_COLOR),
        (155, "Loan balance", LOAN_COLOR),
        (285, "Repaid (cumulative)", REPAID_COLOR),
    )
    for position, label, color in legend_positions:
        legend_layers.extend(
            [
                alt.Chart(alt.Data(values=[{}]))
                .mark_point(filled=True, size=85, color=color, opacity=0.3)
                .encode(x=alt.value(position), y=alt.value(-37)),
                alt.Chart(alt.Data(values=[{}]))
                .mark_rule(color=color, strokeWidth=1)
                .encode(
                    x=alt.value(position - 7),
                    x2=alt.value(position + 7),
                    y=alt.value(-37),
                ),
                alt.Chart(alt.Data(values=[{}]))
                .mark_text(
                    align="left",
                    baseline="middle",
                    color="#444444",
                    fontSize=10,
                )
                .encode(
                    x=alt.value(position + 11),
                    y=alt.value(-37),
                    text=alt.value(label),
                ),
            ]
        )

    median_values = [
        {
            "month": month.isoformat(),
            "median": float(median[index]),
            "series": label,
            "Median salary (£)": round(float(summary.salary_median[index]) / 100)
            * 100,
            "Median loan balance (£)": round(
                float(summary.balance_median[index]) / 100
            )
            * 100,
            "Median repaid (£)": round(float(summary.cumulative_median[index]) / 100)
            * 100,
            "Median monthly repayment (£)": round(
                float(summary.monthly_repayment_median[index])
            ),
        }
        for median, _fan_low, _fan_high, _color, label in series
        for index, month in enumerate(summary.months)
    ]
    median_tooltip = [
        alt.Tooltip("month:T", title=None, format="%Y"),
        alt.Tooltip("Median salary (£):Q", format=",.0f"),
        alt.Tooltip("Median loan balance (£):Q", format=",.0f"),
        alt.Tooltip("Median repaid (£):Q", format=",.0f"),
        alt.Tooltip("Median monthly repayment (£):Q", format=",.0f"),
    ]
    median_chart = (
        alt.Chart(alt.Data(values=median_values))
        .mark_line(strokeWidth=1)
        .encode(
            x="month:T",
            y="median:Q",
            color=alt.Color("series:N", scale=color_scale, legend=None),
            tooltip=median_tooltip,
        )
    )

    # Each month is assigned a precomputed matched path. The trajectory selection
    # accepts only the first pointer event after entry, then remains locked until
    # mouseleave clears its store.
    hover_rng = np.random.default_rng(SIMULATION_SEED + 1)
    hover_trajectory_choices = np.concatenate(
        [
            hover_rng.permutation(EXAMPLE_PATHS)
            for _ in range((len(summary.months) + EXAMPLE_PATHS - 1) // EXAMPLE_PATHS)
        ]
    )[: len(summary.months)]
    hover_values = [
        {
            "month": month.isoformat(),
            "trajectory": int(hover_trajectory_choices[index]),
            "Median salary (£)": round(float(summary.salary_median[index]) / 100)
            * 100,
            "Median loan balance (£)": round(
                float(summary.balance_median[index]) / 100
            )
            * 100,
            "Median repaid (£)": round(float(summary.cumulative_median[index]) / 100)
            * 100,
            "Median monthly repayment (£)": round(
                float(summary.monthly_repayment_median[index])
            ),
        }
        for index, month in enumerate(summary.months)
    ]
    nearest_month = alt.selection_point(
        name="hover_month",
        nearest=True,
        on="pointerover[!length(data('hover_trajectory_store'))]",
        fields=["month"],
        empty=False,
        clear="view:mouseleave",
    )
    hover_trajectory = alt.selection_point(
        name="hover_trajectory",
        nearest=True,
        on="pointerover",
        fields=["trajectory"],
        empty=False,
        clear="view:mouseleave",
    )
    hover_active = "length(data('hover_trajectory_store')) > 0"
    legend_layers.extend(
        [
            alt.Chart(alt.Data(values=[{}]))
            .mark_rule(color="#555555", strokeWidth=0.75)
            .encode(
                x=alt.value(125),
                x2=alt.value(163),
                y=alt.value(-12),
                opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
            ),
            alt.Chart(alt.Data(values=[{}]))
            .mark_text(
                align="left",
                baseline="middle",
                color="#444444",
                fontSize=10,
            )
            .encode(
                x=alt.value(172),
                y=alt.value(-12),
                text=alt.value("Example life trajectory"),
                opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
            ),
        ]
    )
    example_series = (
        (summary.salary_examples, "Salary"),
        (summary.balance_examples, "Loan balance"),
        (summary.cumulative_examples, "Repaid (cumulative)"),
    )
    example_values = [
        {
            "month": month.isoformat(),
            "value": float(paths[month_index, path_index]),
            "series": label,
            "trajectory": path_index,
        }
        for paths, label in example_series
        for path_index in range(paths.shape[1])
        for month_index, month in enumerate(summary.months)
    ]
    example_chart = (
        alt.Chart(alt.Data(values=example_values))
        .mark_line(strokeWidth=0.65)
        .encode(
            x="month:T",
            y="value:Q",
            color=alt.Color("series:N", scale=color_scale, legend=None),
            detail=["series:N", "trajectory:N"],
            opacity=alt.condition(hover_trajectory, alt.value(0.5), alt.value(0)),
        )
    )
    hover_selectors = (
        alt.Chart(alt.Data(values=hover_values))
        .mark_rule(color="#666666", strokeWidth=0.75)
        .encode(
            x="month:T",
            opacity=alt.condition(nearest_month, alt.value(0.25), alt.value(0)),
            tooltip=median_tooltip,
        )
        .add_params(nearest_month, hover_trajectory)
    )
    hover_points = median_chart.mark_point(
        filled=True,
        size=65,
        stroke="white",
        strokeWidth=0.75,
    ).encode(
        opacity=alt.condition(nearest_month, alt.value(1), alt.value(0))
    )
    write_off_values = [
        {"month": write_off.isoformat(), "plan": plan}
        for plan, write_off in loan_write_offs
    ]
    write_off_rule = (
        alt.Chart(alt.Data(values=write_off_values))
        .mark_rule(color="#999999", strokeDash=[5, 5], strokeWidth=0.75)
        .encode(x="month:T", detail="plan:N")
    )
    write_off_labels = [
        alt.Chart(
            alt.Data(values=[{"month": write_off.isoformat(), "label": plan}])
        )
        .mark_text(
            align="left",
            baseline="top",
            color="#777777",
            dx=4,
            fontSize=8,
            lineBreak="\n",
            lineHeight=9,
        )
        .encode(x="month:T", y=alt.value(4 + index * 11), text="label:N")
        for index, (plan, write_off) in enumerate(loan_write_offs)
    ]
    chart = (
        alt.layer(
            band_chart,
            write_off_rule,
            *write_off_labels,
            median_chart,
            example_chart,
            hover_points,
            hover_selectors,
            *legend_layers,
        )
        .properties(
            width=500,
            height=500,
            padding={"top": 55, "right": 5, "bottom": 5, "left": 5},
        )
        .configure_view(stroke=None)
        .configure_axis(grid=False, labelColor="#444444", titleColor="#444444")
    )
    return chart.to_dict(validate=False)


st.set_page_config(page_title="Student loan forecast", page_icon="📈", layout="wide")

if "loan_ids" not in st.session_state:
    st.session_state.loan_ids = [0]
if "next_loan_id" not in st.session_state:
    st.session_state.next_loan_id = 1

with st.sidebar:
    st.header("Your assumptions")
    salary = st.number_input(
        "Current gross annual salary (£)",
        0.0,
        value=50_000.0,
        step=1000.0,
        format="%.0f",
    )
    forecast_start = TODAY_MONTH
    st.markdown("**Student loans**")
    loan_defaults = (
        (PLAN_2, 75_000.0, 2019),
        (POSTGRADUATE, 12_000.0, 2023),
        (PLAN_1, 20_000.0, 2011),
        (PLAN_4, 30_000.0, 2020),
        (PLAN_5, 45_000.0, 2026),
    )
    entered_loans: list[Loan] = []
    remove_loan_id: int | None = None
    for position, loan_id in enumerate(st.session_state.loan_ids):
        default_plan, default_balance, default_graduation_year = loan_defaults[
            min(position, len(loan_defaults) - 1)
        ]
        saved_plan = st.session_state.get(f"loan_plan_{loan_id}", default_plan)
        with st.expander(
            f"Loan {position + 1} · {saved_plan}",
            expanded=position == 0 or loan_id == st.session_state.loan_ids[-1],
        ):
            plan = st.selectbox(
                "Repayment plan",
                PLAN_NAMES,
                index=PLAN_NAMES.index(default_plan),
                key=f"loan_plan_{loan_id}",
            )
            loan_balance = st.number_input(
                "Current loan balance (£)",
                min_value=0.0,
                value=default_balance,
                step=100.0,
                format="%.0f",
                key=f"loan_balance_{loan_id}",
            )
            graduation_year = int(
                st.number_input(
                    "Graduation or course-leaving year",
                    min_value=2007,
                    max_value=forecast_start.year,
                    value=min(default_graduation_year, forecast_start.year),
                    step=1,
                    format="%d",
                    key=f"loan_graduation_{loan_id}",
                    help=(
                        "Used to estimate the April repayment start and write-off year. "
                        "Part-time courses and older Plan 1/4 loans can follow different rules."
                    ),
                )
            )
            loan = Loan(plan, float(loan_balance), date(graduation_year, 7, 1))
            entered_loans.append(loan)
            st.caption(
                f"Repayments assumed from {loan.repayment_start.year} · "
                f"write-off {loan.write_off_month.year}"
            )
            if position > 0 and st.button(
                "Remove this loan",
                key=f"remove_loan_{loan_id}",
                use_container_width=True,
            ):
                remove_loan_id = loan_id

    if remove_loan_id is not None:
        st.session_state.loan_ids.remove(remove_loan_id)
        st.rerun()

    if st.button(
        "＋ Add another loan",
        disabled=len(st.session_state.loan_ids) >= len(PLAN_NAMES),
        use_container_width=True,
    ):
        st.session_state.loan_ids.append(st.session_state.next_loan_id)
        st.session_state.next_loan_id += 1
        st.rerun()

    loans = tuple(entered_loans)
    selected_plans = [loan.plan for loan in loans]
    if len(selected_plans) != len(set(selected_plans)):
        st.error(
            "Add each repayment plan only once. Combine balances that belong to the "
            "same plan, because payroll treats them as one plan balance."
        )
        st.stop()
    expired_loans = [loan.plan for loan in loans if loan.write_off_month <= forecast_start]
    if expired_loans:
        st.error(
            "The estimated write-off date has already passed for: "
            + ", ".join(expired_loans)
            + ". Check the loan's plan and course-leaving year."
        )
        st.stop()
    with st.expander("Economic assumptions"):
        st.caption(
            "These long-run centres are planning assumptions. Historical data fit "
            "the persistence and annual shock size, not these two values."
        )
        future_rpi = st.number_input(
            "Long-run annual RPI (%)",
            min_value=0.0,
            value=2.5,
            step=0.25,
            format="%.2f",
            help=(
                "The mean-reversion centre for simulated RPI. The 2.50% default is "
                "a rounded planning value close to the OBR's 2.40% long-run projection."
            ),
        )
        real_salary_growth = st.number_input(
            "Long-run real salary growth (%)",
            min_value=-5.0,
            max_value=20.0,
            value=1.25,
            step=0.25,
            format="%.2f",
            help=(
                "The mean-reversion centre for salary growth after RPI. The 1.25% "
                "default is slightly below the roughly 1.40% implied by the OBR's "
                "long-run RPI and nominal-earnings projections."
            ),
        )
        nominal_salary_growth = future_rpi + real_salary_growth
        st.caption(
            f"{nominal_salary_growth:.2f}% implied nominal salary growth · "
            f"{future_rpi:.2f}% RPI"
        )
    purchasing_power_basis = "today's £"
    real_terms = st.toggle(
        f"Purchasing-power view ({purchasing_power_basis})",
        value=False,
        help=(
            "Uses simulated RPI to express chart values and comparisons in pounds "
            "at today's prices."
        ),
    )
    st.divider()
    final_write_off = max(loan.write_off_month for loan in loans)
    payoff_choice = st.selectbox(
        "Pay off all loans in full",
        ["Never", "Now"] + list(range(forecast_start.year, final_write_off.year)),
        help=(
            "Now settles every entered balance immediately. A selected year settles "
            "all remaining balances after that December's interest and required repayments."
        ),
    )

payoff_immediately = payoff_choice == "Now"
payoff_month = (
    None
    if payoff_choice in ("Never", "Now")
    else date(int(payoff_choice), 12, 1)
)
rpi = future_rpi / 100
projection = project(
    salary,
    nominal_salary_growth / 100,
    loans,
    rpi,
    forecast_start,
    payoff_month,
    payoff_immediately,
)
with st.spinner(f"Simulating {SIMULATION_RUNS:,} possible futures..."):
    simulation = simulate(
        salary,
        real_salary_growth / 100,
        loans,
        rpi,
        forecast_start,
        payoff_month,
        payoff_immediately,
        real_terms,
    )
selected_plan_cost = simulation.total_repaid_median
total_balance = sum(loan.balance for loan in loans)
payoff_difference = selected_plan_cost - total_balance
payoff_difference_low = simulation.total_repaid_low - total_balance
payoff_difference_high = simulation.total_repaid_high - total_balance
payoff_answer = payoff_verdict(
    simulation.payoff_worthwhile_probability,
    payoff_immediately,
)
if selected_plan_cost >= total_balance:
    payoff_outcome = "likely savings would be"
else:
    payoff_outcome = "likely losses would be"

st.markdown(
    """
    <style>
    .simulation-metric-label {font-size: 0.75rem; margin-bottom: 0.1rem;}
    .simulation-metric-value {font-size: 1.35rem; line-height: 1.5;}
    .simulation-summary-card {
        background: rgba(221, 221, 221, 0.62);
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 0.4rem;
        padding: 0.65rem 0.8rem;
        height: 6rem;
        box-sizing: border-box;
    }
    .range-errors {
        display: inline-grid; grid-template-rows: 1fr 1fr; margin-left: 0.2em;
        vertical-align: middle; line-height: 1;
    }
    .range-errors sup, .range-errors sub {
        font-size: 0.6em; line-height: 1; position: static;
    }
    .intro-highlight {
        display: inline-block; padding: 0.02rem 0.3rem; border-radius: 0.2rem;
        font-weight: 600; line-height: 1.3;
    }
    .intro-salary {
        color: #2d526a; background: rgba(60, 102, 130, 0.18);
        border: 1px solid rgba(60, 102, 130, 0.32);
    }
    .intro-loan {
        color: #9e2f50; background: rgba(193, 65, 104, 0.16);
        border: 1px solid rgba(193, 65, 104, 0.30);
    }
    .intro-repaid {
        color: #7f520f; background: rgba(200, 135, 45, 0.20);
        border: 1px solid rgba(200, 135, 45, 0.34);
    }
    div[data-testid="stVegaLiteChart"] {
        display: flex; justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

basis = f" ({purchasing_power_basis})" if real_terms else ""
chart_spec = make_simulation_chart(
    simulation,
    tuple((loan.plan, loan.write_off_month) for loan in loans),
    real_terms,
)

intro_column, results_column = st.columns([1, 1.45], gap="large")
with intro_column:
    st.title("Student loan forecast")
    st.markdown(
        "Enter some basic information about your "
        '<span class="intro-highlight intro-salary">salary</span> and '
        '<span class="intro-highlight intro-loan">student loan</span>, and the '
        f"dashboard will simulate {SIMULATION_RUNS:,} possible futures and estimate "
        'your likely lifetime <span class="intro-highlight intro-repaid">repayments</span>. '
        "Inflation and real salary growth follow "
        "separate stochastic processes (mean-reverting AR(1) processes, fitted to historical economic data). "
        "Use the purchasing-power view to express future money at today's prices.",
        unsafe_allow_html=True,
    )

with results_column:
    total_column, payoff_column = st.columns(2, gap="small")
    with total_column:
        st.markdown(
            '<div class="simulation-summary-card">'
            f'<div class="simulation-metric-label">Median total repayments{basis}</div>'
            f'<div class="simulation-metric-value">'
            f'{money_range_html(selected_plan_cost, simulation.total_repaid_low, simulation.total_repaid_high)}'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with payoff_column:
        st.markdown(
            '<div class="simulation-summary-card">'
            '<div class="simulation-metric-label">Worth paying off now?<br>'
            f'<strong>{payoff_answer}</strong>, {payoff_outcome}</div>'
            f'<div class="simulation-metric-value">'
            f'{money_range_html(payoff_difference, payoff_difference_low, payoff_difference_high, signed=True)}'
            '</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div style="height: 1rem;"></div>', unsafe_allow_html=True)
    st.vega_lite_chart(chart_spec, width="content", theme=None)

st.subheader("Assumptions")
loan_summary = "; ".join(
    f"{loan.plan}: {money(loan.balance)}, leaving {loan.graduation_date.year}, "
    f"write-off {loan.write_off_month.year}"
    for loan in loans
)
threshold_summary = "; ".join(
    f"{loan.plan} {money(PLAN_RULES[loan.plan].current_threshold)}"
    for loan in loans
)
st.markdown(
    f"""
The model follows your current salary and loan balances month by month from today, adding interest and subtracting repayments until each loan is paid off or written off.

- **Salary and inflation:** your settings assume {real_salary_growth:.2f}% average annual salary growth above inflation and {future_rpi:.2f}% inflation (RPI). Together, these imply about {nominal_salary_growth:.2f}% annual salary growth in cash terms. The defaults are planning assumptions, not predictions about your career.
- **How growth varies:** real salary growth and RPI inflation are modelled as mean-reverting AR(1) processes around your selected averages. This means unusually high or low growth tends to move back towards those averages over time, while new random changes occur each year. The persistence and variance of those stochastic processes are fitted to historical ONS data.
- **Different possible futures:** the model runs {SIMULATION_RUNS:,} simulations, allowing salary growth and inflation to vary each year. The shaded chart shows the range of simulated outcomes; it is not a guarantee.
- **Repayments:** you pay 9% of earnings above your undergraduate loan threshold, plus a separate 6% above the postgraduate threshold if you have one. Multiple undergraduate loans share the same 9% repayment. Your current annual thresholds are {threshold_summary}.
- **Future loan rules:** the model applies announced rates and threshold changes, then assumes most thresholds rise with inflation. Plan 2's threshold stays frozen through 2029/30, and the postgraduate threshold stays at £21,000. Interest depends on your plan and, for Plan 2, your income. Future interest on Plans 1 and 4 is estimated using inflation alone.
- **Write-off:** any remaining balance is cancelled after the modelled repayment period: 25 years for Plan 1, 30 for Plans 2, 4 and postgraduate, and 40 for Plan 5. Older loans and some course types can have different rules that this model does not cover.
- **Paying off early:** “Now” means paying off today. Choosing a later year clears all remaining loans at the end of that year, after interest and normal repayments. The “Worth paying off now?” answer reflects how often simulated total repayments cost more than clearing your balances now.
- **Today's money:** the purchasing-power view adjusts future amounts for simulated inflation, so you can compare them in {forecast_start.year} pounds.
- **Main limitations:** salary is especially uncertain. Promotions, career breaks, job changes and bonuses can make your actual path very different. The model does not include investment returns, tax effects, future policy changes or other extra repayments.

Use this as a planning estimate. Before clearing a loan, request an exact settlement figure from the [Student Loans Company](https://www.gov.uk/repaying-your-student-loan/make-extra-repayments).
"""
)

with st.expander("How the AR(1) models are fitted"):
    st.markdown(
        """
**What the model means.** AR(1) stands for *first-order autoregression*. It is a
compact statistical description of persistence: if inflation or real-pay growth is
unusually high this year, some fraction of that deviation tends to remain next year,
then fades over time. It is not an economic theory or an ONS forecast. This app fits
the model to ONS history and uses it to create serially correlated paths rather than
unrelated year-by-year draws.

**Where the data come from.** The calibration uses annual observations from
**2000–2025** for two published ONS index series: the RPI All Items Index, series
**CHAW**, and whole-economy Average Weekly Earnings, total pay and seasonally
adjusted, series **KAB9**. Their annual index levels are embedded in the app so that
every deployment uses the same reproducible calibration.

**How the inputs are constructed.** Consecutive index levels are converted into 25
annual growth observations. Historical real earnings growth is calculated by
deflating nominal earnings growth by RPI:
"""
    )
    st.latex(
        r"g^{real}_t = \frac{1 + g^{nominal}_t}{1 + \pi_t} - 1"
    )
    st.markdown(
        "**How it is fitted.** Each growth series is fitted separately by ordinary "
        "least squares using:"
    )
    st.latex(r"x_t = a + \phi x_{t-1} + \varepsilon_t")
    st.markdown(
        rf"""
Here $\phi$ is the estimated one-year persistence: 0 would mean no carry-over from
last year's deviation, while a value closer to 1 would mean shocks fade slowly. The
residuals $\varepsilon_t$ are the one-year changes that persistence does not explain;
their standard deviation supplies the forecast shock size.

This fit produces **RPI persistence of {RPI_PERSISTENCE:.3f}** with annual residual
volatility of **{RPI_INNOVATION_STD * 100:.2f} percentage points**, and **real-earnings
persistence of {REAL_SALARY_PERSISTENCE:.3f}** with residual volatility of
**{REAL_SALARY_INNOVATION_STD * 100:.2f} percentage points**. Volatility is the
sample standard deviation of the fitted one-year residuals.

**How it becomes a forecast.** The historical fit supplies $\phi$ and the shock size
$\sigma$, but the sidebar supplies the forecast's long-run centre. The fitted
intercept is replaced by that selected inflation or real-pay rate, so each path follows:
"""
    )
    st.latex(r"x_t = \mu + \phi(x_{t-1} - \mu) + \sigma z_t")
    st.markdown(
        r"""
where $\mu$ is the selected long-run rate, $\phi$ and $\sigma$ come from the historical
fit, and $z_t$ is a new standard-normal shock each year. Inflation and real-salary
shocks are sampled independently; persistence still makes each series depend on its
own preceding year. Nominal salary growth then combines simulated RPI and simulated
real-salary growth.

The historical observations are embedded in the app, so calibration is reproducible
and does not depend on a live data download. This remains a simplified statistical
model: parameters are treated as constant, the short historical sample includes the
pandemic and inflation shock, and future policy or structural changes are excluded.

[ONS RPI series](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/chaw) ·
[ONS earnings series](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9)
"""
    )

with st.expander("View baseline monthly calculation"):
    running = 0.0
    table = []
    for row in projection.rows:
        running += row.repayment + row.voluntary_payment
        table.append(
            {
                "Year": row.month.year,
                "Annual gross salary": round(row.salary / 100) * 100,
                "Opening balance": round(row.opening_balance / 100) * 100,
                "Annual interest rate": row.interest_rate,
                "Interest added": round(row.interest / 100) * 100,
                "Required repayment": round(row.repayment / 100) * 100,
                "Voluntary payoff": round(row.voluntary_payment / 100) * 100,
                "Cumulative repayments": round(running / 100) * 100,
                "Closing balance": round(row.closing_balance / 100) * 100,
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
