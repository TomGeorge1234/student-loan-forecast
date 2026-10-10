# UK Plan 2 student-loan forecaster

## Launch

```bash
uv pip install -e .
streamlit run student_loan_gui.py
```

Then open the local address printed in the terminal, usually
<http://localhost:8501>. Stop the app with `Ctrl+C`.

Run both commands inside the same activated virtual environment. If you do not
already have one, create and activate it with `uv venv && source .venv/bin/activate`.

## Optional payoff heatmap

Click **Explore salary and loan balance** under **Explore repayment outcomes in a salary × loan amount state space** below the
forecast to generate the heatmap. The square map has a fixed 20 × 20 grid, covering salaries
from £0 to £200,000 and total loan balances from £0 to £150,000.
The map shows median savings from clearing the loan now compared with making
only required repayments, with white at zero savings. Toggle **Show savings vs. paying off now**
off to view median total lifetime repayments instead; switching views uses the existing results.
It uses the current economic and purchasing-power
settings, holding each loan’s share, plan and write-off date fixed. The sidebar
payoff date does not affect this comparison. Hover for probabilities, verdicts and median savings;
the cross marks your current inputs.

The grid runs only on request and retains results in the current session. Changed assumptions hide the old
map until you regenerate it.

Run calculation tests with `python -m unittest`.
