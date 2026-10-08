"""Interactive UK Plan 2 student-loan forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import altair as alt
import numpy as np
import streamlit as st


THRESHOLD = 29_385.0
UPPER_INTEREST_THRESHOLD = 52_885.0
START_MONTH = date.today().replace(day=1)

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


def interest_rate(
    salary: float,
    rpi: float,
    lower: float,
    upper: float,
    cap: float | None = None,
) -> float:
    if salary <= lower:
        extra = 0.0
    elif salary >= upper:
        extra = 0.03
    else:
        extra = 0.03 * (salary - lower) / (upper - lower)
    rate = rpi + extra
    return min(rate, cap) if cap is not None else rate


def add_months(month: date, count: int = 1) -> date:
    """Return the first day of the month `count` months later."""
    index = month.year * 12 + month.month - 1 + count
    return date(index // 12, index % 12 + 1, 1)


def project(
    initial_salary: float,
    salary_growth: float,
    initial_balance: float,
    future_rpi: float,
    write_off_month: date,
    payoff_month: date | None = None,
    payoff_immediately: bool = False,
) -> Projection:
    rows: list[Row] = []
    annual_salary = initial_salary
    balance = initial_balance
    total_repaid = 0.0
    total_interest = 0.0
    month = START_MONTH
    monthly_salary_growth = (1 + salary_growth) ** (1 / 12) - 1

    while month < write_off_month and balance > 0.005:
        tax_year_start = month.year if month.month >= 4 else month.year - 1
        indexed_years = max(0, tax_year_start - 2029)
        lower = THRESHOLD * (1 + future_rpi) ** indexed_years
        upper = UPPER_INTEREST_THRESHOLD * (1 + future_rpi) ** indexed_years
        use_current_rate = month < date(2027, 9, 1)
        rpi = 0.041 if use_current_rate else future_rpi
        cap = 0.06 if use_current_rate else None
        rate = interest_rate(annual_salary, rpi, lower, upper, cap)

        is_now = payoff_immediately and month == START_MONTH
        interest = 0.0 if is_now else balance * ((1 + rate) ** (1 / 12) - 1)
        repayment = 0.0 if is_now else min(
            balance + interest,
            max(0.0, annual_salary - lower) * 0.09 / 12,
        )
        is_payoff_month = payoff_month is not None and month == payoff_month
        voluntary = (
            max(0.0, balance + interest - repayment)
            if is_now or is_payoff_month
            else 0.0
        )
        closing = max(0.0, balance + interest - repayment - voluntary)
        rows.append(
            Row(
                month,
                annual_salary,
                balance,
                rate,
                interest,
                repayment,
                closing,
                voluntary,
            )
        )
        total_repaid += repayment + voluntary
        total_interest += interest
        balance = closing

        if balance <= 0.005:
            outcome = f"Repaid in {month.strftime('%B %Y')}"
            if is_now:
                outcome = "Paid off now"
            elif is_payoff_month:
                outcome = f"Paid off in December {month.year}"
            return Projection(tuple(rows), total_repaid, total_interest, 0.0, outcome)
        annual_salary *= 1 + monthly_salary_growth
        month = add_months(month)

    return Projection(
        tuple(rows),
        total_repaid,
        total_interest,
        balance,
        f"Written off in April {write_off_month.year}",
    )


@st.cache_data(show_spinner=False)
def simulate(
    initial_salary: float,
    mean_real_salary_growth: float,
    initial_balance: float,
    mean_rpi: float,
    write_off_month: date,
    payoff_month: date | None = None,
    payoff_immediately: bool = False,
    real_terms: bool = False,
    runs: int = SIMULATION_RUNS,
) -> SimulationSummary:
    """Simulate mean-reverting inflation and real-salary growth paths."""
    display_end = add_months(write_off_month, 12)
    month_count = (
        (display_end.year - START_MONTH.year) * 12
        + display_end.month
        - START_MONTH.month
        + 1
    )
    months = tuple(add_months(START_MONTH, offset) for offset in range(month_count))
    years = tuple(range(START_MONTH.year, display_end.year + 1))
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

    threshold_multipliers: dict[int, np.ndarray] = {}
    multiplier = np.ones(runs)
    for tax_year in range(2030, write_off_month.year + 1):
        multiplier = multiplier * (1 + annual_rpi[year_indices[tax_year]])
        threshold_multipliers[tax_year] = multiplier.copy()

    salary = np.full(runs, initial_salary, dtype=float)
    loan_balance = np.full(runs, initial_balance, dtype=float)
    cumulative_repaid = np.zeros(runs)
    deflator = np.ones(runs)
    salary_paths = np.empty((month_count, runs))
    balance_paths = np.empty((month_count, runs))
    cumulative_paths = np.empty((month_count, runs))

    for offset, month in enumerate(months):
        year_index = year_indices[month.year]
        loan_active = month < write_off_month

        if loan_active:
            tax_year_start = month.year if month.month >= 4 else month.year - 1
            threshold_multiplier = threshold_multipliers.get(tax_year_start, 1.0)
            lower = THRESHOLD * threshold_multiplier
            upper = UPPER_INTEREST_THRESHOLD * threshold_multiplier
            current_rate_period = month < date(2027, 9, 1)
            rpi_for_interest = (
                np.full(runs, 0.041)
                if current_rate_period
                else annual_rpi[year_index]
            )
            extra_interest = np.clip(
                0.03 * (salary - lower) / (upper - lower), 0.0, 0.03
            )
            interest_rate_values = rpi_for_interest + extra_interest
            if current_rate_period:
                interest_rate_values = np.minimum(interest_rate_values, 0.06)

            is_now = payoff_immediately and month == START_MONTH
            interest = (
                np.zeros(runs)
                if is_now
                else loan_balance * ((1 + interest_rate_values) ** (1 / 12) - 1)
            )
            required_repayment = (
                np.zeros(runs)
                if is_now
                else np.minimum(
                    loan_balance + interest,
                    np.maximum(0.0, salary - lower) * 0.09 / 12,
                )
            )
            is_payoff_month = payoff_month is not None and month == payoff_month
            voluntary_payment = (
                np.maximum(0.0, loan_balance + interest - required_repayment)
                if is_now or is_payoff_month
                else np.zeros(runs)
            )
            loan_balance = np.maximum(
                0.0,
                loan_balance - required_repayment - voluntary_payment + interest,
            )
            monthly_repayment = required_repayment + voluntary_payment
        else:
            loan_balance = np.zeros(runs)
            monthly_repayment = np.zeros(runs)

        display_deflator = deflator if real_terms else 1.0
        cumulative_repaid = cumulative_repaid + monthly_repayment / display_deflator
        salary_paths[offset] = salary / display_deflator
        balance_paths[offset] = loan_balance / display_deflator
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
            np.mean(final_repayments > initial_balance)
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
    write_off_month: date,
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
        }
        for median, _fan_low, _fan_high, _color, label in series
        for index, month in enumerate(summary.months)
    ]
    median_tooltip = [
        alt.Tooltip("month:T", title="Date", format="%b %Y"),
        alt.Tooltip("Median salary (£):Q", format=",.0f"),
        alt.Tooltip("Median loan balance (£):Q", format=",.0f"),
        alt.Tooltip("Median repaid (£):Q", format=",.0f"),
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
    write_off_rule = (
        alt.Chart(alt.Data(values=[{"month": write_off_month.isoformat()}]))
        .mark_rule(color="#999999", strokeDash=[5, 5], strokeWidth=0.75)
        .encode(x="month:T")
    )
    write_off_label = (
        alt.Chart(
            alt.Data(
                values=[
                    {"month": write_off_month.isoformat(), "label": "written\noff"}
                ]
            )
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
        .encode(x="month:T", y=alt.value(4), text="label:N")
    )
    chart = (
        alt.layer(
            band_chart,
            write_off_rule,
            write_off_label,
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

with st.sidebar:
    st.header("Your assumptions")
    salary = st.number_input(
        "Current gross annual salary (£)",
        0.0,
        value=50_000.0,
        step=1000.0,
        format="%.0f",
    )
    balance = st.number_input(
        "Current loan balance (£)",
        0.0,
        value=77_500.0,
        step=100.0,
        format="%.0f",
    )
    with st.expander("Advanced economic assumptions"):
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
    real_terms = st.toggle(
        "Purchasing-power view (today's £)",
        value=False,
        help="Uses assumed RPI to show all chart values and comparisons in today's pounds.",
    )
    st.divider()
    graduation_year = st.number_input(
        "Graduation year",
        min_value=2012,
        max_value=START_MONTH.year,
        value=2019,
        step=1,
        help=(
            "The model assumes you first became due to repay in April of the following "
            "year. A Plan 2 balance is normally written off 30 years after that April."
        ),
    )
    timer_start_year = int(graduation_year) + 1
    write_off_month = date(timer_start_year + 30, 4, 1)
    payoff_choice = st.selectbox(
        "Pay off in full",
        ["Never", "Now"] + list(range(START_MONTH.year, write_off_month.year)),
        help=(
            "Now settles the entered balance immediately. A selected year settles "
            "the remaining balance after that December's interest and required repayment."
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
    balance,
    rpi,
    write_off_month,
    payoff_month,
    payoff_immediately,
)
with st.spinner(f"Simulating {SIMULATION_RUNS:,} possible futures..."):
    simulation = simulate(
        salary,
        real_salary_growth / 100,
        balance,
        rpi,
        write_off_month,
        payoff_month,
        payoff_immediately,
        real_terms,
    )
selected_plan_cost = simulation.total_repaid_median
payoff_difference = selected_plan_cost - balance
payoff_difference_low = simulation.total_repaid_low - balance
payoff_difference_high = simulation.total_repaid_high - balance
payoff_answer = payoff_verdict(
    simulation.payoff_worthwhile_probability,
    payoff_immediately,
)
if selected_plan_cost >= balance:
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

basis = " (today's £)" if real_terms else ""
chart_spec = make_simulation_chart(simulation, write_off_month, real_terms)

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
        "Use the purchasing-power view to express future £'s in today's £'s.",
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
simulation_assumption = (
    f"- **Monte Carlo and AR(1):** {SIMULATION_RUNS:,} paths. An AR(1) is a standard "
    "time-series model in which part of this year's departure from its long-run "
    "average carries into next year, plus a new random shock. The app fits that "
    "persistence and the shock size separately to annual ONS data from 2000–2025: "
    "RPI persistence "
    f"{RPI_PERSISTENCE:.2f} with {RPI_INNOVATION_STD * 100:.2f} percentage-point "
    f"innovation volatility, and real-earnings persistence {REAL_SALARY_PERSISTENCE:.2f} "
    f"with {REAL_SALARY_INNOVATION_STD * 100:.2f} percentage-point volatility. "
    "The sidebar rates are chosen long-run centres, not outputs of this fit. Each "
    "forecast year gets new, independent Gaussian inflation and real-pay shocks; "
    "nominal salary growth combines the resulting RPI and real salary growth. The "
    "continuous fan layers 31 closely spaced point-by-point percentile ranges, from "
    "the central 5% through 95%, so shading fades toward the tails; these are prediction "
    "ranges, not confidence intervals. "
    "[ONS RPI](https://www.ons.gov.uk/economy/inflationandpriceindices/timeseries/chaw) "
    "[ONS earnings](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/earningsandworkinghours/timeseries/kab9)\n"
)
st.markdown(
    f"""
- **Plan and pay basis:** Plan 2, using gross salary before tax and other deductions. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/what-you-pay)
- **Default economic outlook:** 2.50% long-run RPI and 1.25% real salary growth are rounded planning anchors, not fitted historical means. The OBR's July 2026 long-run projection uses 2.4% RPI and 3.8% nominal earnings, implying about 1.4% real earnings relative to RPI; 1.25% is a slightly conservative default. Historical ONS data instead determine the AR(1) persistence and volatility. [OBR](https://obr.uk/frs/fiscal-risks-and-sustainability-july-2026/)
- **Selected outlook:** {real_salary_growth:.2f}% expected real salary growth and {future_rpi:.2f}% expected RPI, implying {nominal_salary_growth:.2f}% long-run nominal salary growth. Recent earnings growth was 3.5% for regular pay and 3.9% for total pay. [ONS](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/averageweeklyearningsingreatbritain/september2026) [OBR forecasts](https://obr.uk/faq/where-can-i-find-your-latest-forecasts/)
{simulation_assumption}- **Repayment rule:** 9% of earnings above the current {money(THRESHOLD)} threshold. [HMRC](https://www.gov.uk/guidance/special-rules-for-student-loans#plan-and-loan-types-and-thresholds)
- **Threshold forecast:** frozen through 2029/30, then uprated using assumed RPI ({future_rpi:.2f}%). [DfE methodology](https://explore-education-statistics.service.gov.uk/methodology/student-loan-forecasts-for-england)
- **Interest rule:** RPI plus 0–3% by income; 2026/27 uses 4.1% RPI and the current 6% cap. [GOV.UK](https://www.gov.uk/guidance/how-interest-is-calculated-plan-2)
- **Write-off:** graduation in {int(graduation_year)} is assumed to make you first due to repay in April {timer_start_year}; any remaining balance is therefore written off in April {write_off_month.year}. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/when-your-student-loan-gets-written-off-or-cancelled)
- **Full payoff:** “Now” settles the current balance immediately. A selected year settles the balance after December's interest and required repayment. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/make-extra-repayments)
- **Payoff verdict:** based on the share of simulations in which projected repayments exceed today's payoff amount: Yes (at least 95%), Probably (67–95%), It's unclear (33–67%), Probably not (5–33%), or No (at most 5%).
- **Today's-money view:** future values are divided by cumulative assumed RPI. ONS treats RPI as a legacy measure. [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/methodologies/calculatingtheretailpricesindex)
- **Timing:** the forecast starts in {START_MONTH.strftime('%B %Y')}, inferred from the system date, and compounds salary, interest and repayments monthly.
- **Important salary caveat:** the strongest—and likely least correct—assumption here is the salary path. It is impossible to know how an individual career will develop: promotions, career breaks, job changes, redundancy, bonuses, working hours and salary ceilings create jumps and structural changes. A stochastic exponential-growth model cannot represent these well, so the salary forecast is frankly likely to be flawed. Sorry!
- **Limitations:** monthly estimate; payroll timing, bonuses, investment returns, liquidity, tax, risk, policy changes, and other voluntary repayments are excluded. Request an exact settlement figure before paying. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/make-extra-repayments)
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
                "Month": row.month.strftime("%Y-%m"),
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
