"""Exercise publication-relevant corruption paths with the real retained run."""
import copy
import json
from pathlib import Path
import unittest
from verify import check

ROOT = Path(__file__).resolve().parent


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / "results.json").read_text())

    def test_actual_run_checks(self):
        check(self.record, ROOT)

    def reject(self, change):
        record = copy.deepcopy(self.record)
        change(record)
        with self.assertRaises(ValueError):
            check(record, ROOT)

    def test_missing_repeat_rejected(self):
        self.reject(lambda r: r["conditions"].pop())

    def test_missing_query_rejected(self):
        self.reject(lambda r: r["conditions"][0]["probabilities"].pop())

    def test_probability_class_swap_rejected(self):
        self.reject(lambda r: r["conditions"][0]["probabilities"][84].reverse())

    def test_nan_rejected(self):
        self.reject(lambda r: r["conditions"][0]["probabilities"][84].__setitem__(1, float("nan")))

    def test_out_of_range_rejected(self):
        self.reject(lambda r: r["conditions"][0]["probabilities"][84].__setitem__(1, 1.1))

    def test_undeclared_label_change_rejected(self):
        self.reject(lambda r: r["conditions"][1]["context_labels"].__setitem__(6, 0))

    def test_wrong_summary_rejected(self):
        self.reject(lambda r: r["summary"].__setitem__("changed_predictions", 15))

    def test_wrong_checkpoint_rejected(self):
        self.reject(lambda r: r.__setitem__("weight_sha256", "0" * 64))

    def test_wrong_protocol_hash_rejected(self):
        self.reject(lambda r: r.__setitem__("protocol_sha256", "0" * 64))

    def test_query_input_change_rejected(self):
        self.reject(lambda r: r["queries"][0].__setitem__("x", -1.19))


if __name__ == "__main__":
    unittest.main()
