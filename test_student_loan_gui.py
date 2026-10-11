"""Regression checks for the UI's paired-view cache."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
from streamlit.testing.v1 import AppTest

from forecast import payoff_grid_views


class RepaymentExplorerCacheTests(unittest.TestCase):
    def test_cached_grid_replays_without_external_layout_effects(self):
        payoff_grid_views.clear()
        self.addCleanup(payoff_grid_views.clear)

        def calculate(*args, **kwargs):
            # Exercise the old progress callback if it leaks into the cache.
            callback = kwargs.get('_progress_callback')
            if callback is not None:
                callback(1.0)
            shape = (2, len(args[1]), len(args[0]))
            return np.full(shape, 0.5), np.zeros(shape)

        app_path = str(Path(__file__).with_name('student_loan_gui.py'))
        with patch('forecast.payoff_grid', side_effect=calculate) as calculate_grid:
            app = AppTest.from_file(app_path).run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(calculate_grid.call_count, 1)
            self.assertEqual(len(app.get('vega_lite_chart')), 2)
            main_spec = json.loads(app.get('vega_lite_chart')[0].proto.spec)
            self.assertEqual(main_spec['width'], main_spec['height'])
            padding = main_spec['padding']
            self.assertEqual(
                main_spec['width'] - padding['left'] - padding['right'],
                main_spec['height'] - padding['top'] - padding['bottom'],
            )
            self.assertEqual(main_spec['autosize'], {'type': 'none', 'contains': 'padding'})
            datasets = app.get('vega_lite_chart')[0].proto.datasets
            example_data = next(d for d in datasets if d.name.startswith('forecast_examples_'))
            examples = pa.ipc.open_stream(example_data.data.data).read_all()
            domain = main_spec['layer'][0]['encoding']['color']['scale']['domain']
            self.assertEqual(set(examples.column('series').to_pylist()), set(domain))


            self.assertFalse(any(b.label == 'Calculate' for b in app.button))
            app.run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(calculate_grid.call_count, 1)

            # Force another cached call after the original layout has disappeared.
            app.session_state['payoff_map'] = None
            app.run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(calculate_grid.call_count, 1)

            for label, value in (
                ('Year-to-year inflation variation (percentage points)', 1.0),
                ('How long inflation changes last (0–0.99)', 0.2),
                ('Year-to-year salary growth variation (percentage points)', 1.5),
                ('How long salary growth changes last (0–0.99)', 0.3),
            ):
                control = next(n for n in app.sidebar.number_input if n.label == label)
                previous_calls = calculate_grid.call_count
                control.set_value(value).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertEqual(calculate_grid.call_count, previous_calls + 1)
            self.assertEqual(calculate_grid.call_args.kwargs['rpi_persistence'], 0.2)
            self.assertEqual(calculate_grid.call_args.kwargs['rpi_innovation_std'], 0.01)
            self.assertEqual(calculate_grid.call_args.kwargs['salary_persistence'], 0.3)
            self.assertEqual(calculate_grid.call_args.kwargs['salary_innovation_std'], 0.015)
            calls_before_toggle = calculate_grid.call_count

            power_toggle = next(
                toggle for toggle in app.toggle
                if toggle.label == "Show amounts in today's money"
            )
            self.assertFalse(any(t.label == "Show amounts in today's money" for t in app.sidebar.toggle))
            power_toggle.set_value(True).run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(calculate_grid.call_count, calls_before_toggle)
            self.assertEqual(len(app.get('vega_lite_chart')), 2)
            self.assertFalse(app.toggle(key='show_payoff_savings').value)
            app.toggle(key='show_payoff_savings').set_value(True).run(timeout=60)
            self.assertFalse(app.exception)
            self.assertEqual(calculate_grid.call_count, calls_before_toggle)
            matrix_spec = app.get('vega_lite_chart')[1].proto.spec
            self.assertIn('Savings from paying off now', matrix_spec)
            main_spec = json.loads(app.get('vega_lite_chart')[0].proto.spec)
            self.assertEqual(main_spec['width'], main_spec['height'])



if __name__ == '__main__':
    unittest.main()
