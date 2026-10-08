from datetime import date
import unittest

import numpy as np

from loan_rules import (
    PLAN_1,
    PLAN_2,
    PLAN_4,
    PLAN_5,
    PLAN_RULES,
    POSTGRADUATE,
    Loan,
    allocate_required_repayments,
    annual_interest_rate,
    threshold_for_tax_year,
)


class RepaymentAllocationTests(unittest.TestCase):
    def test_plan_1_and_plan_2_share_one_nine_percent_deduction(self) -> None:
        salary = np.array([38_400.0])
        balances = np.array([[10_000.0], [20_000.0]])
        active = np.ones_like(balances, dtype=bool)
        thresholds = np.array([[26_900.0], [29_385.0]])

        payments = allocate_required_repayments(
            salary, balances, active, thresholds, (PLAN_1, PLAN_2)
        )

        self.assertAlmostEqual(float(payments.sum()), 86.25)
        self.assertAlmostEqual(float(payments[0, 0]), 18.6375)
        self.assertAlmostEqual(float(payments[1, 0]), 67.6125)

    def test_paid_off_lower_plan_does_not_lower_the_threshold(self) -> None:
        salary = np.array([38_400.0])
        balances = np.array([[0.0], [20_000.0]])
        active = np.array([[False], [True]])
        thresholds = np.array([[26_900.0], [29_385.0]])

        payments = allocate_required_repayments(
            salary, balances, active, thresholds, (PLAN_1, PLAN_2)
        )

        self.assertAlmostEqual(float(payments[0, 0]), 0.0)
        self.assertAlmostEqual(float(payments[1, 0]), 67.6125)

    def test_postgraduate_repayment_stacks_with_plan_2(self) -> None:
        salary = np.array([30_000.0])
        balances = np.array([[20_000.0], [10_000.0]])
        active = np.ones_like(balances, dtype=bool)
        thresholds = np.array([[29_385.0], [21_000.0]])

        payments = allocate_required_repayments(
            salary, balances, active, thresholds, (PLAN_2, POSTGRADUATE)
        )

        self.assertAlmostEqual(float(payments[0, 0]), 4.6125)
        self.assertAlmostEqual(float(payments[1, 0]), 45.0)
        self.assertAlmostEqual(float(payments.sum()), 49.6125)


class PlanRuleTests(unittest.TestCase):
    def test_plan_2_threshold_is_frozen_then_uprated(self) -> None:
        rpi = {year: 0.03 for year in range(2026, 2032)}
        rule = PLAN_RULES[PLAN_2]

        self.assertEqual(threshold_for_tax_year(rule, 2029, rpi), 29_385.0)
        self.assertAlmostEqual(
            float(threshold_for_tax_year(rule, 2030, rpi)),
            29_385.0 * 1.03,
        )

    def test_other_threshold_policies_are_distinct(self) -> None:
        rpi = {year: 0.03 for year in range(2026, 2032)}
        self.assertEqual(
            threshold_for_tax_year(PLAN_RULES[PLAN_1], 2027, rpi),
            28_005.0,
        )
        self.assertAlmostEqual(
            float(threshold_for_tax_year(PLAN_RULES[PLAN_1], 2028, rpi)),
            28_005.0 * 1.03,
        )
        self.assertAlmostEqual(
            float(threshold_for_tax_year(PLAN_RULES[PLAN_5], 2027, rpi)),
            25_000.0 * 1.03,
        )
        self.assertEqual(
            threshold_for_tax_year(PLAN_RULES[POSTGRADUATE], 2031, rpi),
            21_000.0,
        )

    def test_plan_5_write_off_is_forty_years_after_repayment_start(self) -> None:
        loan = Loan(PLAN_5, 40_000.0, date(2026, 7, 1))
        self.assertEqual(loan.repayment_start, date(2027, 4, 1))
        self.assertEqual(loan.write_off_month, date(2067, 4, 1))

    def test_modern_write_off_terms_differ_by_plan(self) -> None:
        graduation = date(2020, 7, 1)
        expected_years = {
            PLAN_1: 2046,
            PLAN_2: 2051,
            PLAN_4: 2051,
            POSTGRADUATE: 2051,
        }
        for plan, expected_year in expected_years.items():
            with self.subTest(plan=plan):
                self.assertEqual(
                    Loan(plan, 10_000.0, graduation).write_off_month.year,
                    expected_year,
                )

    def test_plan_2_current_interest_varies_with_income(self) -> None:
        rule = PLAN_RULES[PLAN_2]
        salaries = np.array([20_000.0, 60_000.0])
        thresholds = np.array([29_385.0, 29_385.0])
        rates = annual_interest_rate(
            rule,
            salaries,
            np.array([0.041, 0.041]),
            thresholds,
            date(2026, 10, 1),
        )
        np.testing.assert_allclose(rates, np.array([0.041, 0.06]))

    def test_plan_specific_interest_rules(self) -> None:
        salary = np.array([40_000.0])
        threshold = np.array([25_000.0])
        future_rpi = np.array([0.025])

        plan_5_rate = annual_interest_rate(
            PLAN_RULES[PLAN_5], salary, future_rpi, threshold, date(2028, 1, 1)
        )
        postgraduate_rate = annual_interest_rate(
            PLAN_RULES[POSTGRADUATE],
            salary,
            future_rpi,
            np.array([21_000.0]),
            date(2028, 1, 1),
        )

        self.assertAlmostEqual(float(plan_5_rate[0]), 0.025)
        self.assertAlmostEqual(float(postgraduate_rate[0]), 0.055)


if __name__ == "__main__":
    unittest.main()
