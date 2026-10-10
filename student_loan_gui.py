"""Interactive multi-plan UK student-loan forecaster."""

from __future__ import annotations

import hashlib
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
)

from forecast import (
    TODAY_MONTH, SIMULATION_RUNS, SIMULATION_SEED, FAN_COVERAGES, EXAMPLE_PATHS,
    RPI_PERSISTENCE, RPI_INNOVATION_STD,
    REAL_SALARY_PERSISTENCE, REAL_SALARY_INNOVATION_STD,
    SimulationSummary, project, simulate, payoff_grid,
)

# Fixed samples from Seaborn's perceptually uniform crest and flare colour maps.
SALARY_COLOR = "#3C6682"  # crest
LOAN_COLOR = "#C14168"  # flare
REPAID_COLOR = "#C8872D"


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


def chart_data_revision(
    summary: SimulationSummary,
    loan_write_offs: tuple[tuple[str, date], ...],
    payoff_month: date | None,
) -> str:
    """Return a stable version for the data embedded in a chart spec.

    Streamlit transports named Vega-Lite datasets separately from the JSON spec
    and expects their names to change with their contents. Keeping a fixed name
    can leave Vega displaying its previous dataset after a Streamlit rerun.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(repr((summary.months, loan_write_offs, payoff_month)).encode())
    for values in (
        summary.monthly_repayment_median,
        summary.salary_median,
        summary.salary_fan_low,
        summary.salary_fan_high,
        summary.balance_median,
        summary.balance_fan_low,
        summary.balance_fan_high,
        summary.cumulative_median,
        summary.cumulative_fan_low,
        summary.cumulative_fan_high,
        summary.salary_examples,
        summary.balance_examples,
        summary.cumulative_examples,
    ):
        contiguous = np.ascontiguousarray(values)
        digest.update(contiguous.dtype.str.encode())
        digest.update(repr(contiguous.shape).encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def make_simulation_chart(
    summary: SimulationSummary,
    loan_write_offs: tuple[tuple[str, date], ...],
    real_terms: bool = False,
    payoff_month: date | None = None,
) -> dict:
    revision = chart_data_revision(summary, loan_write_offs, payoff_month)
    dataset_names = {
        name: f"forecast_{name}_{revision}"
        for name in ("bands", "medians", "examples", "hover")
    }
    # Quarterly points are sufficient for the background curves at this chart
    # width. Keep monthly medians/tooltips and points around balance-clearing
    # events so sampling does not smooth over a payoff or write-off.
    plot_indices = set(range(0, len(summary.months), 3))
    plot_indices.add(len(summary.months) - 1)
    event_months = {month for _plan, month in loan_write_offs}
    if payoff_month is not None:
        event_months.add(payoff_month)
    for index, month in enumerate(summary.months):
        if month in event_months:
            plot_indices.update(
                range(max(0, index - 1), min(len(summary.months), index + 2))
            )
    plot_indices = sorted(plot_indices)
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
            "low": round(float(low[index]), 2),
            "high": round(float(high[index]), 2),
            "series": label,
            "band": band_index,
        }
        for _median, fan_low, fan_high, _color, label in series
        for band_index, (low, high) in enumerate(zip(fan_low, fan_high))
        for index in plot_indices
        for month in (summary.months[index],)
    ]
    color_scale = alt.Scale(
        domain=[item[4] for item in series],
        range=[item[3] for item in series],
    )
    band_chart = (
        alt.Chart(alt.NamedData(name=dataset_names["bands"]))
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
        alt.Tooltip("month:T", title="Month", format="%b %Y"),
        alt.Tooltip("Median salary (£):Q", format=",.0f"),
        alt.Tooltip("Median loan balance (£):Q", format=",.0f"),
        alt.Tooltip("Median repaid (£):Q", format=",.0f"),
        alt.Tooltip("Median monthly repayment (£):Q", format=",.0f"),
    ]
    median_chart = (
        alt.Chart(alt.NamedData(name=dataset_names["medians"]))
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
        on="pointerover",
        fields=["month"],
        empty=False,
        clear="view:mouseleave",
    )
    hover_trajectory = alt.selection_point(
        name="hover_trajectory",
        nearest=True,
        on="pointerover[!length(data('hover_trajectory_store'))]",
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
            "value": round(float(paths[month_index, path_index]), 2),
            "series": label,
            "trajectory": path_index,
        }
        for paths, label in example_series
        for path_index in range(paths.shape[1])
        for month_index in plot_indices
        for month in (summary.months[month_index],)
    ]
    example_chart = (
        alt.Chart(alt.NamedData(name=dataset_names["examples"]))
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
        alt.Chart(alt.NamedData(name=dataset_names["hover"]))
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
    # Attach numeric datasets after serializing the chart structure. This avoids
    # Altair walking every record and lets the median line/points share one table.
    spec = chart.to_dict(validate=False)
    spec.setdefault("datasets", {}).update(
        {
            dataset_names["bands"]: band_values,
            dataset_names["medians"]: median_values,
            dataset_names["examples"]: example_values,
            dataset_names["hover"]: hover_values,
        }
    )
    return spec


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
    .st-key-assumptions [data-testid="stMarkdownContainer"] p,
    .st-key-assumptions [data-testid="stMarkdownContainer"] li {
        font-size: 0.8rem; line-height: 1.55;
    }
    .st-key-payoff-explorer {
        margin-top: 1.2rem;
        border-radius: 0.8rem;
    }
    .payoff-explorer-title {
        font-size: 1.05rem; font-weight: 600; margin: 0 0 0.3rem;
    }
    .payoff-explorer-description {
        font-size: 0.9rem; line-height: 1.5; margin: 0; opacity: 0.8;
    }
    .st-key-payoff-explorer button {
        border-radius: 0.6rem; min-height: 3rem; font-weight: 600;
    }
    .st-key-payoff-explorer button[kind="primary"] {
        background-color: #3c6682; border-color: #3c6682; color: white;
    }
    .st-key-payoff-explorer button[kind="primary"]:hover {
        background-color: #2d526a; border-color: #2d526a;
    }
    .st-key-payoff-explorer button[kind="primary"]:disabled {
        opacity: 0.5;
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
    payoff_month,
)

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

with st.container():
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

salary_max = 200_000.0
balance_max = 150_000.0
resolution = 20
grid_salaries = tuple(np.linspace(0, salary_max, resolution))
grid_balances = tuple(np.linspace(0, balance_max, resolution))
grid_inputs = (
    grid_salaries, grid_balances, loans, real_salary_growth / 100,
    rpi, forecast_start, real_terms,
)
with st.container(border=True, key="payoff-explorer"):
    question_column, button_column = st.columns([1.7, 1], gap="large", vertical_alignment="center")
    with question_column:
        st.markdown(
            '<h3 class="payoff-explorer-title">Explore repayment outcomes in a [salary × loan] state-space</h3>',
            unsafe_allow_html=True,
        )
    with button_column:
        generate_heatmap = st.button(
            "Explore salary and loan balance", type="primary",
            use_container_width=True, disabled=total_balance <= 0,
        )
    if total_balance <= 0:
        st.caption("Enter a positive loan balance to explore repayment savings.")
if generate_heatmap:
    progress = st.progress(0, text="Simulating salary and loan combinations… 0%")
    def update_heatmap_progress(fraction):
        progress.progress(fraction, text=f"Simulating salary and loan combinations… {fraction:.0%}")
    try:
        result = payoff_grid(*grid_inputs, _progress_callback=update_heatmap_progress)
        progress.progress(1.0, text="Heatmap complete")
        st.session_state.payoff_map = (grid_inputs, result)
    finally:
        progress.empty()

saved_map = st.session_state.get("payoff_map")
if saved_map is not None and saved_map[0] != grid_inputs:
    st.info("Assumptions have changed. Click the button above to update the heatmap.")
elif saved_map is not None:
    probabilities, savings = saved_map[1]
    heatmap_units = "today's £" if real_terms else "£"
    show_savings = st.toggle("Show savings vs. paying off now", value=True)
    lifetime_repayments = savings + np.array(grid_balances)[:, None]
    salary_edges = np.concatenate((
        [0], (np.array(grid_salaries[:-1]) + grid_salaries[1:]) / 2, [salary_max],
    ))
    balance_edges = np.concatenate((
        [0], (np.array(grid_balances[:-1]) + grid_balances[1:]) / 2, [balance_max],
    ))
    cells = [
        {
            "Salary": annual_salary, "Loan": balance,
            "x0": salary_edges[column], "x1": salary_edges[column + 1],
            "y0": balance_edges[row], "y1": balance_edges[row + 1],
            "Probability": probabilities[row, column],
            "Median savings": savings[row, column],
            "Lifetime repayments": lifetime_repayments[row, column],
            "Repayments display": f"£{lifetime_repayments[row, column]:,.0f}",
            "Savings display": f"{'−' if savings[row, column] < 0 else ''}£{abs(savings[row, column]):,.0f}",
            "Verdict": "No loan" if balance == 0 else payoff_verdict(probabilities[row, column]),
        }
        for row, balance in enumerate(grid_balances)
        for column, annual_salary in enumerate(grid_salaries)
    ]
    savings_limit = max(float(np.max(np.abs(savings))), 1.0)
    if show_savings:
        colour_field = "Median savings:Q"
        legend_title = f"Median savings ({heatmap_units})"
        colour_scale = alt.Scale(
            domain=[-savings_limit, 0, savings_limit],
            range=[LOAN_COLOR, "#f7f7f7", SALARY_COLOR],
        )
        colour_description = "Blue means savings; pink means paying now costs more; white means no saving."
    else:
        colour_field = "Lifetime repayments:Q"
        legend_title = f"Lifetime repayments ({heatmap_units})"
        colour_scale = alt.Scale(
            domain=[0, max(float(np.max(lifetime_repayments)), 1.0)],
            range=["#f7f7f7", SALARY_COLOR],
        )
        colour_description = "Darker blue means higher median lifetime repayments with required repayments only."
    currency_tick = (
        "(datum.value < 0 ? '−' : '') + '£' + "
        "(abs(datum.value) >= 1000 ? format(abs(datum.value) / 1000, '~g') + 'k' "
        ": format(abs(datum.value), '~g'))"
    )
    heatmap = alt.Chart(alt.Data(values=cells)).mark_rect().encode(
        x=alt.X("x0:Q", title=f"Current gross annual salary ({heatmap_units})",
                scale=alt.Scale(domain=[0, salary_max], nice=False), axis=alt.Axis(labelExpr=currency_tick)),
        x2="x1:Q",
        y=alt.Y("y0:Q", title=f"Total loan balance ({heatmap_units})",
                scale=alt.Scale(domain=[0, balance_max], nice=False), axis=alt.Axis(labelExpr=currency_tick)),
        y2="y1:Q",
        color=alt.Color(
            colour_field, title=legend_title, scale=colour_scale,
            legend=alt.Legend(labelExpr=currency_tick, orient="top", gradientLength=300),
        ),
        tooltip=[
            alt.Tooltip("Salary:Q", title=f"Salary ({heatmap_units})", format=",.0f"),
            alt.Tooltip("Loan:Q", title=f"Loan ({heatmap_units})", format=",.0f"),
            alt.Tooltip("Probability:Q", title="Chance paying off now saves money", format=".1%"),
            alt.Tooltip("Verdict:N"),
            alt.Tooltip("Repayments display:N", title=f"Median lifetime repayments{basis}"),
            alt.Tooltip("Savings display:N", title=f"Median saving by paying off now{basis}"),
        ],
    )
    if salary <= salary_max and total_balance <= balance_max:
        marker = alt.Chart(alt.Data(values=[{"salary": salary, "balance": total_balance}])).mark_point(
            shape="cross", size=180, color="black", strokeWidth=2, tooltip=False,
        ).encode(x="salary:Q", y="balance:Q")
        heatmap = heatmap + marker
    else:
        st.caption("Your current salary or balance is outside the heatmap ranges.")
    st.altair_chart(
        heatmap.properties(
            width=500, height=500,
            autosize=alt.AutoSizeParams(type="pad", contains="content"),
        ),
        width="content",
    )
    st.caption(
        f"{resolution} × {resolution} grid · {SIMULATION_RUNS:,} futures per point · "
        f"{'Today’s purchasing power' if real_terms else 'Nominal cash amounts'} · "
        "Black cross marks your inputs when in range. Hover over any cell to see "
        "the chance paying off now saves money, median lifetime repayments and the median amount saved. "
        "Negative savings mean paying now costs more. "
        + colour_description
    )

loan_summary = "; ".join(
    f"{loan.plan}: {money(loan.balance)}, leaving {loan.graduation_date.year}, "
    f"write-off {loan.write_off_month.year}"
    for loan in loans
)
threshold_summary = "; ".join(
    f"{loan.plan} {money(PLAN_RULES[loan.plan].current_threshold)}"
    for loan in loans
)
with st.container(key="assumptions"):
    with st.expander("Assumptions", expanded=False):
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
