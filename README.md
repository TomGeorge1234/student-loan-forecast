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

The sidebar groups inputs into blue **Salary assumptions**, red **Student loan**,
and yellow **Repayment assumptions** boxes. **Other economic assumptions** contains
the long-run RPI rate, annual shock volatility (in percentage points), and AR(1)
persistence. Volatility and persistence default to the historical fit; changing
these coefficients refreshes both forecast views and the heatmap. Salary growth,
volatility, and persistence are inside the collapsed **Salary growth assumptions**
section. In the model, nominal salary growth is RPI plus real salary growth.
The purchasing-power toggle is on the main page above the results.

## Salary and loan comparison

The heatmap loads automatically in its own grey section below the main square forecast,
with a loading indicator while it is being calculated. The square map has a fixed 20 × 20 grid, covering salaries
from £0 to £200,000 and total loan balances from £0 to £150,000.
The map defaults to median lifetime repayments. The alternative view shows savings
from clearing the loan now compared with your selected repayment schedule, with
white at zero savings. Switch the toggle between **Lifetime repayments** and **Worth paying off now?** beneath the matrix
to choose the map view; switching views uses the existing results.
It uses the current economic and purchasing-power
settings, holding each loan’s share, plan and write-off date fixed. The sidebar
payoff date applies to this comparison and changing it regenerates the map. Hover for probabilities, verdicts and median savings;
the outlined blue cross marks your current inputs.

The grid loads automatically and retains results in the current session.
Changed assumptions automatically regenerate the map.
Both cash and purchasing-power results are calculated together and cached, so switching
the purchasing-power toggle reuses the existing simulations and heatmap.

Run calculation tests with `python -m unittest`.
