"""Interactive UK Plan 2 student-loan forecaster."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO

import matplotlib.dates as mdates
import streamlit as st
from matplotlib import style as mpl_style
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator


THRESHOLD = 29_385.0
UPPER_INTEREST_THRESHOLD = 52_885.0
START_MONTH = date.today().replace(day=1)

# Fixed samples from Seaborn's perceptually uniform crest and flare colour maps.
SALARY_COLOR = "#3C6682"  # crest
LOAN_COLOR = "#C14168"  # flare
REPAID_COLOR = LOAN_COLOR

ASSUMPTION_PRESETS = {
    "Cautious": (3.0, 3.0),
    "Central": (4.0, 3.0),
    "Strong career progression": (5.0, 3.0),
}
DEFAULT_PRESET = "Central"

mpl_style.use(
    {
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 1.0,
        "axes.edgecolor": "#444444",
        "axes.facecolor": "none",
        "axes.labelcolor": "#444444",
        "axes.titlecolor": "#444444",
        "text.color": "#444444",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.facecolor": "none",
    }
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


def repayment_value(projection: Projection, inflation: float = 0.0) -> float:
    return sum(
        (row.repayment + row.voluntary_payment)
        / (1 + inflation) ** (offset / 12)
        for offset, row in enumerate(projection.rows)
    )


def projection_series(projection: Projection, inflation: float = 0.0):
    months, salaries, balances, cumulative = [], [], [], []
    running_total = 0.0
    for offset, row in enumerate(projection.rows):
        deflator = (1 + inflation) ** (offset / 12)
        months.append(row.month)
        salaries.append(row.salary / deflator)
        balances.append(row.closing_balance / deflator)
        running_total += (row.repayment + row.voluntary_payment) / deflator
        cumulative.append(running_total)
    return months, salaries, balances, cumulative


def money(value: float) -> str:
    return f"£{value:,.0f}"


def compact_pounds(value: float, _position=None) -> str:
    if abs(value) >= 1_000_000:
        scaled, suffix = value / 1_000_000, "m"
    elif abs(value) >= 1_000:
        scaled, suffix = value / 1_000, "k"
    else:
        return money(value)
    digits = 0 if float(scaled).is_integer() else 1
    return f"£{scaled:.{digits}f}{suffix}"


def apply_assumption_preset() -> None:
    salary_growth, future_rpi = ASSUMPTION_PRESETS[
        st.session_state["outlook_preset"]
    ]
    st.session_state["salary_growth_input"] = salary_growth
    st.session_state["future_rpi_input"] = future_rpi


def make_figure(
    projection: Projection,
    write_off_month: date,
    salary_growth: float,
    inflation: float = 0.0,
) -> Figure:
    months, salaries, balances, cumulative = projection_series(projection, inflation)
    display_end = add_months(write_off_month, 12)
    chart_months = list(months)
    monthly_salary_growth = (1 + salary_growth) ** (1 / 12) - 1
    salary = projection.rows[-1].salary
    next_month = add_months(months[-1])
    offset = len(months)
    while next_month <= display_end:
        salary *= 1 + monthly_salary_growth
        deflator = (1 + inflation) ** (offset / 12)
        chart_months.append(next_month)
        salaries.append(salary / deflator)
        balances.append(0.0)
        cumulative.append(cumulative[-1])
        next_month = add_months(next_month)
        offset += 1

    figure = Figure(figsize=(4.5, 4.5))
    FigureCanvasAgg(figure)
    axis = figure.add_axes((0.17, 0.15, 0.76, 0.68))
    axis.plot(
        chart_months,
        salaries,
        color=SALARY_COLOR,
        linewidth=1.5,
        label="Salary",
        zorder=10,
    )
    axis.plot(
        chart_months,
        balances,
        color=LOAN_COLOR,
        linewidth=1.5,
        label="Loan balance",
        zorder=10,
    )
    axis.fill_between(
        chart_months,
        cumulative,
        color=REPAID_COLOR,
        alpha=0.2,
        linewidth=0,
        label="Repaid (cumulative)",
        zorder=1,
    )
    axis.axvline(
        write_off_month,
        color="0.6",
        linewidth=0.75,
        linestyle="--",
        zorder=5,
    )
    axis.axhline(0, color="0.6", linewidth=0.5, alpha=0.65, zorder=0)
    money_basis = "£ (today's money)" if inflation else "£ (nominal)"
    axis.set(xlabel="date", ylabel=money_basis, xlim=(months[0], display_end))
    axis.set_ylim(bottom=0)
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.yaxis.set_major_formatter(FuncFormatter(compact_pounds))
    axis.set_box_aspect(1)
    offset = 2 * 72 / 25.4
    axis.spines["bottom"].set_position(("outward", offset))
    axis.spines["left"].set_position(("outward", offset))

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.97),
        ncols=3,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=0.8,
    )
    return figure


st.set_page_config(page_title="Student loan forecast", page_icon="📈", layout="wide")
st.title("Student loan forecast")

with st.sidebar:
    st.header("Your assumptions")
    salary = st.number_input(
        "Current gross annual salary (£)", 0.0, value=50_000.0, step=1_000.0
    )
    balance = st.number_input("Current loan balance (£)", 0.0, value=77_471.69, step=1_000.0)
    if "salary_growth_input" not in st.session_state:
        st.session_state["salary_growth_input"] = ASSUMPTION_PRESETS[DEFAULT_PRESET][0]
    if "future_rpi_input" not in st.session_state:
        st.session_state["future_rpi_input"] = ASSUMPTION_PRESETS[DEFAULT_PRESET][1]
    preset_names = list(ASSUMPTION_PRESETS)
    with st.container(border=True):
        preset = st.selectbox(
            "Career and inflation outlook",
            preset_names,
            index=preset_names.index(DEFAULT_PRESET),
            key="outlook_preset",
            on_change=apply_assumption_preset,
            help=(
                "Cautious assumes 3% salary growth; Central 4%; Strong career "
                "progression 5%. All use 3% future RPI. These are scenarios, not forecasts."
            ),
        )
        st.caption(
            f"{st.session_state['salary_growth_input']:.2f}% salary growth · "
            f"{st.session_state['future_rpi_input']:.2f}% RPI"
        )
        with st.expander("Fine-tune assumptions"):
            salary_growth = st.number_input(
                "Annual salary growth (%)",
                min_value=0.0,
                step=0.25,
                key="salary_growth_input",
                help="Nominal annual growth. Selecting a preset resets this value.",
            )
            future_rpi = st.number_input(
                "Future annual RPI (%)",
                min_value=0.0,
                step=0.25,
                key="future_rpi_input",
                help="Long-run RPI assumption. Selecting a preset resets this value.",
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
    salary_growth / 100,
    balance,
    rpi,
    write_off_month,
    payoff_month,
    payoff_immediately,
)
inflation = rpi if real_terms else 0.0
selected_plan_cost = repayment_value(projection, inflation)
difference = abs(balance - selected_plan_cost)
if payoff_immediately:
    payoff_answer = "Selected"
elif balance < selected_plan_cost:
    payoff_answer = "Yes"
elif selected_plan_cost < balance:
    payoff_answer = "No"
else:
    payoff_answer = "Either"

st.markdown(
    """
    <style>
    [data-testid="stMetricLabel"] p {font-size: 0.75rem;}
    [data-testid="stMetricValue"] {font-size: 1.35rem;}
    .payoff-verdict {font-size: 0.9rem; margin: 0.35rem 0 0.9rem; color: #444;}
    </style>
    """,
    unsafe_allow_html=True,
)

basis = " (today's £)" if real_terms else ""
st.metric(f"Total repaid{basis}", money(repayment_value(projection, inflation)))

figure = make_figure(projection, write_off_month, salary_growth / 100, inflation)
svg = BytesIO()
figure.savefig(svg, format="svg", dpi=600, bbox_inches="tight")
st.image(svg.getvalue().decode("utf-8"), width=500)
figure.clear()
st.markdown(
    f'<p class="payoff-verdict"><strong>Pay off now?</strong> '
    f'{payoff_answer} (save {money(difference)})</p>',
    unsafe_allow_html=True,
)

st.subheader("Assumptions")
st.markdown(
    f"""
- **Plan and pay basis:** Plan 2, using gross salary before tax and other deductions. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/what-you-pay)
- **Selected outlook:** {preset} baseline, currently set to {salary_growth:.2f}% annual salary growth and {future_rpi:.2f}% annual RPI. Recent earnings growth was 3.5% for regular pay and 3.9% for total pay. [ONS](https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/bulletins/averageweeklyearningsingreatbritain/september2026) [OBR forecasts](https://obr.uk/faq/where-can-i-find-your-latest-forecasts/)
- **Repayment rule:** 9% of earnings above the current £29,385 threshold. [HMRC](https://www.gov.uk/guidance/special-rules-for-student-loans#plan-and-loan-types-and-thresholds)
- **Threshold forecast:** frozen through 2029/30, then uprated using assumed RPI ({future_rpi:.2f}%). [DfE methodology](https://explore-education-statistics.service.gov.uk/methodology/student-loan-forecasts-for-england)
- **Interest rule:** RPI plus 0–3% by income; 2026/27 uses 4.1% RPI and the current 6% cap. [GOV.UK](https://www.gov.uk/guidance/how-interest-is-calculated-plan-2)
- **Write-off:** graduation in {int(graduation_year)} is assumed to make you first due to repay in April {timer_start_year}; any remaining balance is therefore written off in April {write_off_month.year}. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/when-your-student-loan-gets-written-off-or-cancelled)
- **Full payoff:** “Now” settles the current balance immediately. A selected year settles the balance after December's interest and required repayment. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/make-extra-repayments)
- **Today's-money view:** future values are divided by cumulative assumed RPI. ONS treats RPI as a legacy measure. [ONS](https://www.ons.gov.uk/economy/inflationandpriceindices/methodologies/calculatingtheretailpricesindex)
- **Timing:** the forecast starts in {START_MONTH.strftime('%B %Y')}, inferred from the system date, and compounds salary, interest and repayments monthly.
- **Limitations:** monthly estimate; payroll timing, bonuses, investment returns, liquidity, tax, risk, policy changes, and other voluntary repayments are excluded. Request an exact settlement figure before paying. [GOV.UK](https://www.gov.uk/repaying-your-student-loan/make-extra-repayments)
"""
)

with st.expander("View monthly calculation"):
    running = 0.0
    table = []
    for row in projection.rows:
        running += row.repayment + row.voluntary_payment
        table.append(
            {
                "Month": row.month.strftime("%Y-%m"),
                "Annual gross salary": row.salary,
                "Opening balance": row.opening_balance,
                "Annual interest rate": row.interest_rate,
                "Interest added": row.interest,
                "Required repayment": row.repayment,
                "Voluntary payoff": row.voluntary_payment,
                "Cumulative repayments": running,
                "Closing balance": row.closing_balance,
            }
        )
    st.dataframe(table, hide_index=True, width="stretch")
