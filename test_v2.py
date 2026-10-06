"""Exercise v2 integrity, failed coverage and no-change reach semantics."""
import copy
import json
from pathlib import Path
import unittest
from metrics_v2 import reach, aggregate, decision
from verify import check

ROOT = Path(__file__).resolve().parent


class V2EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / 'results-v2.json').read_text())

    def reject(self, mutation):
        r = copy.deepcopy(self.record)
        mutation(r)
        with self.assertRaises(ValueError):
            check(r, ROOT)

    def test_actual_all_conditions(self):
        check(self.record, ROOT)
        self.assertEqual(sum(len(m['conditions']) for m in self.record['models'].values()), 51)

    def test_missing_flip(self):
        self.reject(lambda r: r['models']['kumo']['conditions'].pop())

    def test_duplicate_flip(self):
        self.reject(lambda r: r['models']['knn']['conditions'].__setitem__(2, copy.deepcopy(r['models']['knn']['conditions'][1])))

    def test_partial_run(self):
        self.reject(lambda r: r.__setitem__('status', 'aborted'))

    def test_failed_condition(self):
        self.reject(lambda r: r['models']['lr']['conditions'][4].__setitem__('status', 'failed'))

    def test_reused_baseline_is_exact(self):
        self.reject(lambda r: r['models']['kumo']['conditions'][0]['probabilities'][0].__setitem__(1, .8))

    def test_baseline_not_new_execution(self):
        self.reject(lambda r: r['models']['kumo']['conditions'][0].__setitem__('origin', 'v2-execution'))

    def test_class_order_identity(self):
        self.reject(lambda r: r['models']['kumo']['conditions'][2]['raw_columns'].reverse())

    def test_wrong_reach_distance(self):
        self.reject(lambda r: r['models']['kumo']['conditions'][3]['reach'].__setitem__('farthest_changed_distance', 99))

    def test_wrong_aggregate(self):
        self.reject(lambda r: r['summary']['kumo']['changed_predictions'].__setitem__('median', 14))

    def test_false_stop_decision(self):
        self.reject(lambda r: r.__setitem__('decision', 'STOP'))

    def test_undeclared_label(self):
        self.reject(lambda r: r['models']['knn']['conditions'][1]['context_labels'].__setitem__(8, 1))

    def test_changed_default_recipe(self):
        self.reject(lambda r: r['models']['lr']['conditions'][1]['estimator_parameters'].__setitem__('C', 2))

    def test_missing_classical_output(self):
        self.reject(lambda r: r['models']['lr']['conditions'][5]['raw_probabilities'].pop())

    def test_nonfinite_output(self):
        self.reject(lambda r: r['models']['lr']['conditions'][1]['probabilities'][0].__setitem__(1, float('nan')))

    def test_false_checkpoint(self):
        self.reject(lambda r: r.__setitem__('weight_sha256', '0' * 64))

    def test_unknown_schema(self):
        self.reject(lambda r: r.__setitem__('schema_version', 3))

    def test_all_conditions_wrong_default(self):
        def change(r):
            for c in r['models']['lr']['conditions']:
                c['estimator_parameters']['fit_intercept'] = False
        self.reject(change)

    def test_exact_tie_and_empty_reach(self):
        q = [{'x': 0, 'y': 0}, {'x': 3, 'y': 4}]
        original = [[.5, .5], [.4, .6]]
        result = reach(original, [[.5, .5], [.6, .4]], q, q[0])
        self.assertEqual(result['changed_predictions'], 1)
        self.assertAlmostEqual(result['max_probability_change'], .2)
        self.assertEqual(result['farthest_changed_distance'], 5)
        unchanged = reach(original, original, q, q[0])
        self.assertIsNone(unchanged['farthest_changed_distance'])
        self.assertEqual(aggregate([unchanged, result])['farthest_changed_distance'], {'median': 2.5, 'max': 5})

    def test_no_distinction_stops(self):
        zero = aggregate([{'changed_predictions': 0, 'max_probability_change': 0, 'farthest_changed_distance': None}] * 16)
        self.assertEqual(decision({k: zero for k in ['kumo', 'knn', 'lr']}), 'STOP')


if __name__ == '__main__':
    unittest.main()
