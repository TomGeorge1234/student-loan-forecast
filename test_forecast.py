from dataclasses import fields
from unittest.mock import patch
from datetime import date
import unittest

import numpy as np

from forecast import TODAY_MONTH, payoff_grid, payoff_grid_views, simulate
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

    def test_grid_uses_selected_payoff_schedule(self):
        payoff_month = date(TODAY_MONTH.year + 2, 12, 1)
        for immediately in (False, True):
            with self.subTest(immediately=immediately):
                probabilities, savings = payoff_grid(
                    (60_000.,), (50_000.,), self.loans, .0125, .025,
                    runs=30, payoff_month=payoff_month,
                    payoff_immediately=immediately,
                )
                full = simulate.__wrapped__(
                    60_000, .0125, self.loans, .025,
                    payoff_month=payoff_month, payoff_immediately=immediately,
                    runs=30,
                )
                self.assertEqual(probabilities[0, 0], full.payoff_worthwhile_probability)
                self.assertAlmostEqual(savings[0, 0], full.total_repaid_median - 50_000)
                if immediately:
                    self.assertEqual(savings[0, 0], 0)

    def test_paired_forecasts_match_separate_views(self):
        args = (60_000., .0125, self.loans, .025)
        paired = simulate.__wrapped__(*args, runs=30, both_views=True)
        for index, real_terms in enumerate((False, True)):
            separate = simulate.__wrapped__(*args, runs=30, real_terms=real_terms)
            for field in fields(separate):
                expected = getattr(separate, field.name)
                actual = getattr(paired[index], field.name)
                if isinstance(expected, np.ndarray):
                    np.testing.assert_array_equal(actual, expected)
                else:
                    self.assertEqual(actual, expected)

    def test_paired_grid_matches_separate_views_and_reuses_cache(self):
        args = ((0., 60_000.), (0., 50_000.), self.loans, .0125, .025)
        payoff_grid_views.clear()
        with patch('forecast.payoff_grid', wraps=payoff_grid) as calculate:
            paired = payoff_grid_views(*args, runs=30)
            cached = payoff_grid_views(*args, runs=30)
            self.assertEqual(calculate.call_count, 1)
        for index, real_terms in enumerate((False, True)):
            expected = payoff_grid(*args, runs=30, real_terms=real_terms)
            for values, cached_values, separate in zip(paired, cached, expected):
                np.testing.assert_array_equal(values[index], separate)
                np.testing.assert_array_equal(cached_values[index], separate)
        payoff_grid_views.clear()

    def test_custom_rpi_coefficients_affect_forecast_and_grid(self):
        args = (60_000., .0125, self.loans, .025)
        default = simulate.__wrapped__(*args, runs=30)
        deterministic_rpi = simulate.__wrapped__(
            *args, runs=30, rpi_innovation_std=0., rpi_persistence=0.,
        )
        self.assertFalse(np.array_equal(default.salary_fan_high, deterministic_rpi.salary_fan_high))
        sticky_rpi = simulate.__wrapped__(
            *args, runs=30, rpi_innovation_std=.03, rpi_persistence=.9,
        )
        independent_rpi = simulate.__wrapped__(
            *args, runs=30, rpi_innovation_std=.03, rpi_persistence=0.,
        )
        self.assertFalse(np.array_equal(sticky_rpi.salary_median, independent_rpi.salary_median))
        probabilities, savings = payoff_grid_views(
            (60_000.,), (50_000.,), self.loans, .0125, .025, runs=30,
            rpi_persistence=.9, rpi_innovation_std=.03,
        )
        self.assertEqual(probabilities[0, 0, 0], sticky_rpi.payoff_worthwhile_probability)
        self.assertAlmostEqual(savings[0, 0, 0], sticky_rpi.total_repaid_median - 50_000)

    def test_custom_salary_coefficients_affect_forecast_and_grid(self):
        args = (60_000., .0125, self.loans, .025)
        deterministic = simulate.__wrapped__(
            *args, runs=30, rpi_innovation_std=0., salary_innovation_std=0.,
        )
        np.testing.assert_array_equal(deterministic.salary_fan_high[-1], deterministic.salary_median)
        custom = simulate.__wrapped__(
            *args, runs=30, salary_persistence=.9, salary_innovation_std=.03,
        )
        independent = simulate.__wrapped__(
            *args, runs=30, salary_persistence=0., salary_innovation_std=.03,
        )
        self.assertFalse(np.array_equal(custom.salary_median, independent.salary_median))
        probabilities, savings = payoff_grid_views(
            (60_000.,), (50_000.,), self.loans, .0125, .025, runs=30,
            salary_persistence=.9, salary_innovation_std=.03,
        )
        self.assertEqual(probabilities[0, 0, 0], custom.payoff_worthwhile_probability)
        self.assertAlmostEqual(savings[0, 0, 0], custom.total_repaid_median - 50_000)

    def test_zero_loan_proportions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive loan balance'):
            payoff_grid((50_000.,), (10_000.,),
                                      (Loan(PLAN_2, 0, date(2019, 7, 1)),), .01, .025)


if __name__ == '__main__':
    unittest.main()
