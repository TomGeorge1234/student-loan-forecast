from datetime import date
import unittest

import numpy as np

from forecast import TODAY_MONTH, payoff_grid, simulate
from loan_rules import Loan, PLAN_2, POSTGRADUATE


class PayoffGridTests(unittest.TestCase):
    def setUp(self):
        self.loans = (
            Loan(PLAN_2, 40_000, date(2019, 7, 1)),
            Loan(POSTGRADUATE, 10_000, date(2023, 7, 1)),
        )

    def test_grid_matches_full_forecast_and_scales_multiple_loans(self):
        for real_terms in (False, True):
            with self.subTest(real_terms=real_terms):
                probabilities, savings = payoff_grid(
                    (0., 60_000.), (0., 100_000.), self.loans, .0125, .025,
                    TODAY_MONTH, real_terms, runs=30,
                )
                scaled = tuple(Loan(loan.plan, loan.balance * 2, loan.graduation_date)
                               for loan in self.loans)
                full = simulate.__wrapped__(
                    60_000, .0125, scaled, .025, TODAY_MONTH,
                    real_terms=real_terms, runs=30,
                )
                self.assertEqual(probabilities.shape, (2, 2))
                self.assertEqual(probabilities[1, 1], full.payoff_worthwhile_probability)
                self.assertAlmostEqual(savings[1, 1], full.total_repaid_median - 100_000)
                np.testing.assert_array_equal(probabilities[0], 0)
                np.testing.assert_array_equal(savings[0], 0)
                self.assertEqual(probabilities[1, 0], 0)
                self.assertEqual(savings[1, 0], -100_000)

    def test_zero_loan_proportions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive loan balance'):
            payoff_grid((50_000.,), (10_000.,),
                                      (Loan(PLAN_2, 0, date(2019, 7, 1)),), .01, .025)


if __name__ == '__main__':
    unittest.main()
