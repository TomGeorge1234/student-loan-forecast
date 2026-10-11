"""Interactive multi-plan UK student-loan forecaster."""

from __future__ import annotations

import hashlib
from datetime import date

import altair as alt
import numpy as np
import streamlit as st

from heatmap import smooth_heatmap_image

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
    TODAY_MONTH,
    SIMULATION_RUNS,
    SIMULATION_SEED,
    FAN_COVERAGES,
    EXAMPLE_PATHS,
    RPI_PERSISTENCE,
    RPI_INNOVATION_STD,
    REAL_SALARY_PERSISTENCE,
    REAL_SALARY_INNOVATION_STD,
    SimulationSummary,
    project,
    simulate,
    payoff_grid_views,
)

# Fixed samples from Seaborn's perceptually uniform crest and flare colour maps.
SALARY_COLOR = "#3C6682"  # crest
LOAN_COLOR = "#C14168"  # flare
REPAID_COLOR = "#C8872D"


MATRIX_SIZE = 350

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
            "Annual salary",
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
            "Total repaid",
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
        .mark_rect(color="#f0f4f7", opacity=0.9, cornerRadius=6)
        .encode(
            x=alt.value(35),
            x2=alt.value(400),
            y=alt.value(-57),
            y2=alt.value(-16),
        ),
        alt.Chart(alt.Data(values=[{}]))
        .mark_text(
            align="center", baseline="middle", color="#333333",
            fontSize=11, fontWeight=600,
        )
        .encode(
            x=alt.value(217.5), y=alt.value(-43),
            text=alt.value("Your lifetime repayment forecast"),
        ),
    ]
    legend_positions = (
        (50, -25, "Annual salary", SALARY_COLOR),
        (170, -25, "Loan balance", LOAN_COLOR),
        (290, -25, "Total repaid", REPAID_COLOR),
    )
    for position, vertical_position, label, color in legend_positions:
        legend_layers.extend(
            [
                alt.Chart(alt.Data(values=[{}]))
                .mark_point(filled=True, size=85, color=color, opacity=0.3)
                .encode(x=alt.value(position), y=alt.value(vertical_position)),
                alt.Chart(alt.Data(values=[{}]))
                .mark_rule(color=color, strokeWidth=1)
                .encode(
                    x=alt.value(position - 7),
                    x2=alt.value(position + 7),
                    y=alt.value(vertical_position),
                ),
                alt.Chart(alt.Data(values=[{}]))
                .mark_text(
                    align="left",
                    baseline="middle",
                    color="#444444",
                    fontSize=9,
                )
                .encode(
                    x=alt.value(position + 11),
                    y=alt.value(vertical_position),
                    text=alt.value(label),
                ),
            ]
        )

    median_values = [
        {
            "month": month.isoformat(),
            "median": float(median[index]),
            "series": label,
            "Median salary (£)": round(float(summary.salary_median[index]) / 100) * 100,
            "Median loan balance (£)": round(float(summary.balance_median[index]) / 100)
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
            "Median salary (£)": round(float(summary.salary_median[index]) / 100) * 100,
            "Median loan balance (£)": round(float(summary.balance_median[index]) / 100)
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
                y=alt.value(-7),
                opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
            ),
            alt.Chart(alt.Data(values=[{}]))
            .mark_text(
                align="left",
                baseline="middle",
                color="#444444",
                fontSize=9,
            )
            .encode(
                x=alt.value(172),
                y=alt.value(-7),
                text=alt.value("Example life trajectory"),
                opacity=alt.condition(hover_active, alt.value(1), alt.value(0)),
            ),
        ]
    )
    example_series = (
        (summary.salary_examples, series[0][4]),
        (summary.balance_examples, series[1][4]),
        (summary.cumulative_examples, series[2][4]),
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
    ).encode(opacity=alt.condition(nearest_month, alt.value(1), alt.value(0)))
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
        alt.Chart(alt.Data(values=[{"month": write_off.isoformat(), "label": plan}]))
        .mark_text(
            align="left",
            baseline="top",
            color="#777777",
            dx=4,
            fontSize=7.2,
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
            # 519² is 70% of the previous 620² figure area. Equal padding
            # totals keep the 435 × 435 plotting area square too.
            width=519,
            height=519,
            autosize=alt.AutoSizeParams(type="none", contains="padding"),
            padding={"top": 60, "right": 42, "bottom": 24, "left": 42},
        )
        .configure_view(stroke=None)
        .configure_axis(
            grid=False, labelColor="#444444", titleColor="#444444",
            labelFontSize=9, titleFontSize=9.9,
        )
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

forecast_start = TODAY_MONTH
with st.sidebar:
    st.header("Your assumptions")
    with st.container(border=True, key="salary-assumptions"):
        st.markdown("### Salary assumptions")
        salary = st.number_input(
            "Annual salary before tax (£)",
            0.0,
            value=50_000.0,
            step=1000.0,
            format="%.0f",
        )
        with st.expander("Salary growth assumptions", expanded=False):
            real_salary_growth = st.number_input(
                "Annual salary growth above inflation (%)",
                min_value=-5.0,
                max_value=20.0,
                value=1.25,
                step=0.25,
                format="%.2f",
                help=(
                    "Your average annual pay rise above inflation. For example, 1.25% real growth "
                    "plus 2.5% inflation gives 3.75% annual salary growth in this model."
                ),
            )
            salary_volatility = st.number_input(
                "Year-to-year salary growth variation (percentage points)",
                min_value=0.0, max_value=30.0,
                value=float(REAL_SALARY_INNOVATION_STD * 100),
                step=0.25, format="%.2f",
                help="Standard deviation of new annual real-salary growth shocks. Higher values widen the possible salary paths; zero removes random salary-growth shocks.",
            )
            salary_persistence = st.number_input(
                "How long salary growth changes last (0–0.99)",
                min_value=0.0, max_value=0.99,
                value=float(REAL_SALARY_PERSISTENCE),
                step=0.05, format="%.3f",
                help="Higher values make salary-growth shocks persist longer. Zero makes each year's shock independent of the previous year.",
            )
            st.caption("Real growth is the increase above inflation. In this model: nominal salary growth = RPI inflation + real salary growth.")
            salary_growth_summary = st.empty()
            st.caption("Volatility and persistence default to the historical earnings fit.")
    with st.container(border=True, key="student-loan-assumptions"):
        st.markdown("### Student loan")
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
                    f"Repayments start in {loan.repayment_start.year} · "
                    f"Remaining balance cancelled in {loan.write_off_month.year}"
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
        expired_loans = [
            loan.plan for loan in loans if loan.write_off_month <= forecast_start
        ]
        if expired_loans:
            st.error(
                "The estimated write-off date has already passed for: "
                + ", ".join(expired_loans)
                + ". Check the loan's plan and course-leaving year."
            )
            st.stop()
    with st.container(border=True, key="repayment-assumptions"):
        st.markdown("### Repayment assumptions")
        st.caption(
            "By default, you make the required monthly repayments and no more until your loans "
            "are repaid or written off. No voluntary payment in full is scheduled."
        )
        final_write_off = max(loan.write_off_month for loan in loans)
        payoff_choice = st.selectbox(
            "Optional: schedule repayment in full",
            ["Never", "Now"] + list(range(forecast_start.year, final_write_off.year)),
            format_func=lambda choice: (
                "Monthly repayments only (default)" if choice == "Never"
                else "Pay remaining balance now" if choice == "Now"
                else f"Pay remaining balance in December {choice}"
            ),
            help=(
                "Now settles every entered balance immediately. A selected year settles "
                "all remaining balances after that December's interest and required repayments."
            ),
        )
    with st.expander(
        "Other economic assumptions", expanded=False, key="economic-assumptions"
    ):
        future_rpi = st.number_input(
            "Average annual inflation (RPI, %)",
            min_value=0.0,
            value=2.5,
            step=0.25,
            format="%.2f",
            help=(
                "The average inflation rate used in the forecast. RPI is the Retail Prices "
                "Index, which affects loan interest and the value of future money."
            ),
        )
        rpi_volatility = st.number_input(
            "Year-to-year inflation variation (percentage points)",
            min_value=0.0,
            max_value=20.0,
            value=float(RPI_INNOVATION_STD * 100),
            step=0.25,
            format="%.2f",
            help="Standard deviation of new annual RPI shocks. Higher values produce a wider range of inflation outcomes; zero removes random RPI shocks.",
        )
        rpi_persistence = st.number_input(
            "How long inflation changes last (0–0.99)",
            min_value=0.0,
            max_value=0.99,
            value=float(RPI_PERSISTENCE),
            step=0.05,
            format="%.3f",
            help="The AR(1) coefficient. Higher values make inflation shocks persist longer; zero makes each year's shock independent of the previous year.",
        )
        st.caption("Volatility and persistence default to the fit from historical RPI data.")
    nominal_salary_growth = future_rpi + real_salary_growth
    salary_growth_summary.caption(
        f"Your settings: {future_rpi:.2f}% RPI + {real_salary_growth:.2f}% real growth "
        f"= {nominal_salary_growth:.2f}% nominal salary growth per year."
    )

payoff_immediately = payoff_choice == "Now"
payoff_month = (
    None if payoff_choice in ("Never", "Now") else date(int(payoff_choice), 12, 1)
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
simulation_views = simulate(
    salary,
    real_salary_growth / 100,
    loans,
    rpi,
    forecast_start,
    payoff_month,
    payoff_immediately,
    both_views=True,
    rpi_persistence=rpi_persistence,
    rpi_innovation_std=rpi_volatility / 100,
    salary_persistence=salary_persistence,
    salary_innovation_std=salary_volatility / 100,
)

st.markdown(
    """
    <style>
    html {font-size: 90% !important;}
    .stMainBlockContainer {
        max-width: 960px; padding: 2rem 2rem 3rem;
    }
    [data-testid="stSidebar"] {
        background: #e2e4e7;
        border-right: 1px solid #cdd1d6;
    }
    [data-testid="stSidebar"] h2 {font-size: 1.25rem; padding-top: 0;}
    h1 {font-size: 2rem !important; letter-spacing: -0.055rem; padding: 0 !important;}
    [data-testid="stExpander"] {border-radius: 0.65rem;}
    button, [data-baseweb="input"], [data-baseweb="select"] > div {
        border-radius: 0.5rem !important;
    }
    [data-testid="stSidebar"] h3 {font-size: 1rem; padding: 0 0 0.5rem;}
    .st-key-salary-assumptions {
        background: rgba(60, 102, 130, 0.12); border-left: 4px solid #3c6682 !important;
    }
    .st-key-student-loan-assumptions {
        background: rgba(193, 65, 104, 0.10); border-left: 4px solid #c14168 !important;
    }
    .st-key-repayment-assumptions {
        background: rgba(200, 135, 45, 0.14); border-left: 4px solid #c8872d !important;
    }
    .st-key-economic-assumptions details {
        background: rgba(255, 255, 255, 0.3);
        border: 1px solid #cdd1d6;
        border-left: 4px solid #8b929b;
        border-radius: 0.65rem;
    }
    .st-key-economic-assumptions summary {
        padding: 0.9rem 1rem;
    }
    .st-key-economic-assumptions summary [data-testid="stMarkdownContainer"] p {
        font-size: 1rem; font-weight: 600; line-height: 1.4;
    }
    .st-key-economic-assumptions [data-testid="stExpanderDetails"] {
        padding: 0.25rem 1rem 1rem;
    }
    .st-key-matrix-companion {
        background: #ededed;
        color: #252b31;
        border: 1px solid #dedede;
        border-radius: 0.65rem;
        padding: 1.25rem;
        box-sizing: border-box;
    }
    .st-key-matrix-view-toggle [data-testid="stMarkdownContainer"] p {
        font-size: 0.75rem; line-height: 1.3;
    }
    .st-key-matrix-view-toggle [data-testid="stCheckbox"] {
        transform: scale(0.85); transform-origin: center;
    }
    .dashboard-kicker {
        color: #3c6682; font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.12rem; text-transform: uppercase; margin-bottom: 0.25rem;
    }
    .dashboard-intro {font-size: 0.9rem; opacity: 0.7; margin: 0 0 0.3rem;}
    .simulation-summary-grid {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem; margin: 0.5rem 0 1rem;
    }
    .simulation-metric-label {font-size: 0.85rem; color: #555d65; line-height: 1.5; margin-bottom: 0.5rem;}
    .simulation-metric-note {font-size: 0.8rem; color: #555d65; line-height: 1.5; margin-top: 0.5rem;}
    .simulation-metric-value {font-size: 1.65rem; font-weight: 600; line-height: 1.25;}
    .simulation-summary-card {
        background: #ededed;
        color: #252b31;
        border: 1px solid #dedede;
        border-radius: 0.65rem;
        padding: 1rem 1.25rem;
        min-width: 0;
        min-height: 8rem;
        box-sizing: border-box;
    }
    @media (max-width: 640px) {
        .stMainBlockContainer {padding: 1rem;}
        h1 {font-size: 1.65rem !important;}
        .simulation-metric-value {font-size: 1.3rem;}
        .simulation-summary-grid {grid-template-columns: 1fr;}
        .simulation-summary-card {padding: 1rem;}
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
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="dashboard-kicker">UK student loans · Lifetime outlook</div>',
    unsafe_allow_html=True,
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

purchasing_power_basis = "today's £"
real_terms = st.toggle(
    f"Show amounts in today's money",
    value=False,
    help=(
        "Uses simulated RPI to express chart values and comparisons in pounds "
        "at today's prices."
    ),
)

simulation = simulation_views[int(real_terms)]
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
    payoff_outcome = "Estimated savings vs your repayment plan"
else:
    payoff_outcome = "Negative savings mean paying now costs more"


basis = f" ({purchasing_power_basis})" if real_terms else ""
chart_spec = make_simulation_chart(
    simulation,
    tuple((loan.plan, loan.write_off_month) for loan in loans),
    real_terms,
    payoff_month,
)


st.markdown(
    '<div class="simulation-summary-grid">'
    '<div class="simulation-summary-card">'
    f'<div class="simulation-metric-label">Median total lifetime repayments{basis}</div>'
    f'<div class="simulation-metric-value">'
    f"{money_range_html(selected_plan_cost, simulation.total_repaid_low, simulation.total_repaid_high)}"
    '</div><div class="simulation-metric-note">Includes monthly repayments and any scheduled payoff</div></div>'
    '<div class="simulation-summary-card">'
    '<div class="simulation-metric-label">Worth paying off now?</div>'
    f'<div class="simulation-metric-value">'
    f"{money_range_html(payoff_difference, payoff_difference_low, payoff_difference_high, signed=True)}"
    f'</div><div class="simulation-metric-note"><strong>{payoff_answer}</strong> · {payoff_outcome}</div>'
    "</div></div>",
    unsafe_allow_html=True,
)

with st.container(horizontal=True, horizontal_alignment="center"):
    st.vega_lite_chart(chart_spec, width="content", theme=None)
st.caption("Lines show the median forecast; shaded bands show uncertainty. Hover to see monthly values.")
st.divider()

with st.container(key="matrix-companion"):
    st.markdown("**How lifetime repayments vary by salary and loan amount**")
    matrix_results = st.container()

salary_max = 200_000.0
balance_max = 150_000.0
resolution = 20
grid_salaries = tuple(np.linspace(0, salary_max, resolution))
grid_balances = tuple(np.linspace(0, balance_max, resolution))
grid_inputs = (
    grid_salaries,
    grid_balances,
    loans,
    real_salary_growth / 100,
    rpi,
    forecast_start,
    payoff_month,
    payoff_immediately,
    rpi_persistence,
    rpi_volatility / 100,
    salary_persistence,
    salary_volatility / 100,
)
with matrix_results:
    if total_balance <= 0:
        st.caption("Enter a loan balance above £0 to see this comparison.")
    saved_map = st.session_state.get("payoff_map")
    if (
        total_balance > 0
        and (saved_map is None or saved_map[0] != grid_inputs)
    ):
        with st.spinner("Simulating salary and loan combinations…"):
            result = payoff_grid_views(
                *grid_inputs[:8], rpi_persistence=rpi_persistence,
                rpi_innovation_std=rpi_volatility / 100,
                salary_persistence=salary_persistence,
                salary_innovation_std=salary_volatility / 100,
            )
        st.session_state.payoff_map = (grid_inputs, result)

    saved_map = st.session_state.get("payoff_map")
    if (
        saved_map is not None
        and saved_map[0] == grid_inputs
    ):
        probability_views, savings_views = saved_map[1]
        probabilities = probability_views[int(real_terms)]
        savings = savings_views[int(real_terms)]
        heatmap_units = "today's £" if real_terms else "£"
        show_savings = st.session_state.get("show_payoff_savings", False)
        lifetime_repayments = savings + np.array(grid_balances)[:, None]
        salary_edges = np.concatenate(
            (
                [0],
                (np.array(grid_salaries[:-1]) + grid_salaries[1:]) / 2,
                [salary_max],
            )
        )
        balance_edges = np.concatenate(
            (
                [0],
                (np.array(grid_balances[:-1]) + grid_balances[1:]) / 2,
                [balance_max],
            )
        )
        cells = [
            {
                "Salary": annual_salary,
                "Loan": balance,
                "x0": salary_edges[column],
                "x1": salary_edges[column + 1],
                "y0": balance_edges[row],
                "y1": balance_edges[row + 1],
                "Probability": probabilities[row, column],
                "Median savings": savings[row, column],
                "Lifetime repayments": lifetime_repayments[row, column],
                "Repayments display": f"£{lifetime_repayments[row, column]:,.0f}",
                "Savings display": f"{'−' if savings[row, column] < 0 else ''}£{abs(savings[row, column]):,.0f}",
                "Verdict": "No loan"
                if balance == 0
                else payoff_verdict(probabilities[row, column]),
            }
            for row, balance in enumerate(grid_balances)
            for column, annual_salary in enumerate(grid_salaries)
        ]
        savings_limit = max(float(np.max(np.abs(savings))), 1.0)
        if show_savings:
            colour_field = "Median savings:Q"
            legend_title = f"Savings from paying off now ({heatmap_units})"
            colour_scale = alt.Scale(
                domain=[-savings_limit, 0, savings_limit],
                range=[LOAN_COLOR, "#f7f7f7", SALARY_COLOR],
            )
        else:
            colour_field = "Lifetime repayments:Q"
            legend_title = f"Lifetime repayments ({heatmap_units})"
            colour_scale = alt.Scale(
                domain=[0, max(float(np.max(lifetime_repayments)), 1.0)],
                range=["#f7f7f7", SALARY_COLOR],
            )
        currency_tick = (
            "(datum.value < 0 ? '−' : '') + '£' + "
            "(abs(datum.value) >= 1000 ? format(abs(datum.value) / 1000, '~g') + 'k' "
            ": format(abs(datum.value), '~g'))"
        )
        heatmap = (
            alt.Chart(alt.Data(values=cells))
            .mark_rect(fillOpacity=0, strokeWidth=0)
            .encode(
                x=alt.X(
                    "x0:Q",
                    title="Annual salary before tax (£)",
                    scale=alt.Scale(domain=[0, salary_max], nice=False),
                    axis=alt.Axis(labelExpr=currency_tick),
                ),
                x2="x1:Q",
                y=alt.Y(
                    "y0:Q",
                    title="Total loan balance (£)",
                    scale=alt.Scale(domain=[0, balance_max], nice=False),
                    axis=alt.Axis(labelExpr=currency_tick),
                ),
                y2="y1:Q",
                color=alt.Color(
                    colour_field,
                    title=legend_title,
                    scale=colour_scale,
                    legend=alt.Legend(
                        labelExpr=currency_tick, orient="none", direction="horizontal",
                        gradientLength=200, gradientThickness=8,
                        legendX=(MATRIX_SIZE - 200) / 2, legendY=-48,
                        titleAnchor="middle", titleFontSize=8, labelFontSize=8
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Salary:Q", title=f"Salary ({heatmap_units})", format=",.0f"
                    ),
                    alt.Tooltip(
                        "Loan:Q", title=f"Loan ({heatmap_units})", format=",.0f"
                    ),
                    alt.Tooltip(
                        "Probability:Q",
                        title="Chance paying off now saves money",
                        format=".1%",
                    ),
                    alt.Tooltip("Verdict:N"),
                    alt.Tooltip(
                        "Repayments display:N",
                        title=f"Median lifetime repayments{basis}",
                    ),
                    alt.Tooltip(
                        "Savings display:N",
                        title=f"Savings from paying off now{basis}",
                    ),
                ],
            )
        )
        surface = smooth_heatmap_image(
            savings if show_savings else lifetime_repayments,
            tuple(colour_scale.domain), tuple(colour_scale.range), size=MATRIX_SIZE,
        )
        smooth_surface = alt.Chart(alt.Data(values=[{"image": surface}])).mark_image(
            width=MATRIX_SIZE, height=MATRIX_SIZE, tooltip=False,
        ).encode(x=alt.value(MATRIX_SIZE / 2), y=alt.value(MATRIX_SIZE / 2), url="image:N")
        heatmap = smooth_surface + heatmap
        if salary <= salary_max and total_balance <= balance_max:
            marker_position = alt.Chart(
                alt.Data(values=[{"salary": salary, "balance": total_balance}])
            ).encode(x="salary:Q", y="balance:Q")
            marker_halo = marker_position.mark_point(
                shape="cross",
                size=130,
                color="white",
                strokeWidth=4,
                opacity=1,
                tooltip=False,
            )
            marker = marker_position.mark_point(
                shape="cross",
                size=130,
                color="#2d526a",
                strokeWidth=1.5,
                opacity=1,
                tooltip=False,
            )
            heatmap = heatmap + marker_halo + marker
        else:
            st.caption("Your current salary or balance is outside the heatmap ranges.")
        with st.container(horizontal=True, horizontal_alignment="center"):
            st.altair_chart(
                heatmap.properties(
                    width=MATRIX_SIZE,
                    height=MATRIX_SIZE,
                    background="#ededed",
                    padding={"top": 55, "right": 5, "bottom": 5, "left": 5},
                    autosize=alt.AutoSizeParams(type="pad", contains="content"),
                ).configure_axis(labelFontSize=9, titleFontSize=9.9)
                .configure_legend(labelFontSize=9, titleFontSize=9.9),
                width="content",
            )

        with st.container(
            horizontal=True, horizontal_alignment="center", vertical_alignment="center",
            gap="small", key="matrix-view-toggle",
        ):
            st.markdown("Lifetime repayments", width="content")
            st.toggle(
                "Worth paying off now?", value=False, key="show_payoff_savings",
                help="Compare estimated savings from paying now with your selected repayment schedule. Negative savings mean paying now costs more.",
                label_visibility="collapsed", width="content",
            )
            st.markdown("Worth paying off now?", width="content")


loan_summary = "; ".join(
    f"{loan.plan}: {money(loan.balance)}, leaving {loan.graduation_date.year}, "
    f"write-off {loan.write_off_month.year}"
    for loan in loans
)
threshold_summary = "; ".join(
    f"{loan.plan} {money(PLAN_RULES[loan.plan].current_threshold)}" for loan in loans
)
with st.container(key="assumptions"):
    with st.expander("Assumptions", expanded=False):
        st.markdown(
            f"""
The model follows your current salary and loan balances month by month from today, adding interest and subtracting repayments until each loan is paid off or written off.

- **Salary and inflation:** your settings assume {real_salary_growth:.2f}% average annual salary growth above inflation and {future_rpi:.2f}% inflation (RPI). Together, these imply about {nominal_salary_growth:.2f}% annual salary growth in cash terms. The defaults are planning assumptions, not predictions about your career.
- **How growth varies:** real salary growth and RPI inflation are modelled as mean-reverting AR(1) processes around your selected averages. This means unusually high or low growth tends to move back towards those averages over time, while new random changes occur each year. Your selected real-salary persistence is {salary_persistence:.3f}, with annual shock volatility of {salary_volatility:.2f} percentage points. The defaults are fitted to historical ONS data. RPI defaults to the historical fit, with your selected persistence of {rpi_persistence:.3f} and annual shock volatility of {rpi_volatility:.2f} percentage points.
- **Different possible futures:** the model runs {SIMULATION_RUNS:,} simulations, allowing salary growth and inflation to vary each year. The shaded chart shows the range of simulated outcomes; it is not a guarantee.
- **Repayments:** you pay 9% of earnings above your undergraduate loan threshold, plus a separate 6% above the postgraduate threshold if you have one. Multiple undergraduate loans share the same 9% repayment. Your current annual thresholds are {threshold_summary}.
- **Future loan rules:** the model applies announced rates and threshold changes, then assumes most thresholds rise with inflation. Plan 2's threshold stays frozen through 2029/30, and the postgraduate threshold stays at £21,000. Interest depends on your plan and, for Plan 2, your income. Future interest on Plans 1 and 4 is estimated using inflation alone.
- **Write-off:** any remaining balance is cancelled after the modelled repayment period: 25 years for Plan 1, 30 for Plans 2, 4 and postgraduate, and 40 for Plan 5. Older loans and some course types can have different rules that this model does not cover.
- **Paying off early:** “Now” means paying off today. Choosing a later year clears all remaining loans at the end of that year, after interest and normal repayments. The “Worth paying off now?” answer reflects how often simulated total repayments cost more than clearing your balances now.
- **Today's money:** the purchasing-power view adjusts future amounts for simulated inflation, so you can compare them in {forecast_start.year} pounds.
- **Salary and loan comparison:** the second graph varies your current salary and total loan balance while keeping your loan plans, balance proportions, and repayment schedule fixed. Its default view shows median lifetime repayments. The alternative shows estimated savings from clearing the balance now; negative values mean paying now costs more. Colours are smoothly interpolated between simulated grid points, while hover values show the nearest simulated point.
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
            st.latex(r"g^{real}_t = \frac{1 + g^{nominal}_t}{1 + \pi_t} - 1")
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

        **How it becomes a forecast.** The historical fit supplies the default $\phi$ and shock size
        $\sigma$. The sidebar supplies the long-run centre and lets you override RPI and real-salary
        persistence and volatility. Your selected real-salary values are **{salary_persistence:.3f}**
        and **{salary_volatility:.2f} percentage points**. Your selected RPI values are **{rpi_persistence:.3f}**
        and **{rpi_volatility:.2f} percentage points**. The fitted
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

with st.expander("Monthly breakdown (average growth assumptions)"):
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
