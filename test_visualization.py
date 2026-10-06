"""Check the built page against raw probabilities, including ties and no reach."""
import json
import math
from pathlib import Path
import re
import unittest
from visualization import overlays, rgb, luminance, mark_ink

ROOT = Path(__file__).resolve().parent


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((ROOT / 'results-v2.json').read_text(encoding='utf-8'))
        cls.page = (ROOT / 'index.html').read_text(encoding='utf-8')
        cls.drawn = json.loads(re.search(r'<script id="overlays" type="application/json">(.*?)</script>', cls.page, re.S).group(1))

    def test_all_rendered_marks_and_radii_from_probabilities(self):
        for key, model in self.record['models'].items():
            base = model['conditions'][0]['probabilities']
            for index, condition in enumerate(model['conditions']):
                with self.subTest(model=key, condition=index):
                    probabilities = condition['probabilities']
                    classes = [0 if a >= b else 1 for a, b in probabilities]
                    expected = [i for i, ((a, b), (c, d)) in enumerate(zip(base, probabilities, strict=True))
                                if (0 if a >= b else 1) != (0 if c >= d else 1)]
                    drawn = self.drawn[key][index]
                    self.assertEqual(drawn['classes'], classes)
                    self.assertEqual(drawn['changed_cells'], expected)
                    if not expected:
                        self.assertIsNone(drawn['ring'])
                    else:
                        point = self.record['context'][index - 1]
                        farthest = max(math.dist((q['x'], q['y']), (point['x'], point['y']))
                                       for i in expected for q in [self.record['queries'][i]])
                        self.assertEqual(drawn['ring']['cx'], 120 + 80 * point['x'])
                        self.assertEqual(drawn['ring']['cy'], 120 - 80 * point['y'])
                        self.assertAlmostEqual(drawn['ring']['radius_units'], farthest)
                        self.assertAlmostEqual(drawn['ring']['radius_px'], 80 * farthest)

    def test_every_boundary_separates_adjacent_opposite_classes(self):
        for key, conditions in self.drawn.items():
            for index, drawn in enumerate(conditions):
                raw = self.record['models'][key]['conditions'][index]['probabilities']
                expected = []
                for row in range(13):
                    for col in range(13):
                        i = row * 13 + col
                        prediction = raw[i][1] > raw[i][0]
                        x, y = 24 + col * 16, 216 - row * 16
                        if col < 12 and prediction != (raw[i + 1][1] > raw[i + 1][0]):
                            expected.append([x + 8, y - 8, x + 8, y + 8])
                        if row < 12 and prediction != (raw[i + 13][1] > raw[i + 13][0]):
                            expected.append([x - 8, y - 8, x + 8, y - 8])
                self.assertEqual(drawn['boundary_edges'], expected)

    def test_all_marks_have_three_to_one_contrast_against_fill(self):
        for key, conditions in self.drawn.items():
            for index, drawn in enumerate(conditions):
                probabilities = self.record['models'][key]['conditions'][index]['probabilities']
                for i in drawn['changed_cells']:
                    light = luminance(rgb(probabilities[i][1]))
                    ratio = (light + .05) / .05 if drawn['mark_inks'][i] == '#000000' else 1.05 / (light + .05)
                    self.assertGreaterEqual(ratio, 3)
        # Black/white double strokes on boundaries/rings cover the entire fill
        # scale, including antialiasing-independent nominal interior colours.
        for i in range(1001):
            light = luminance(rgb(i / 1000))
            self.assertGreaterEqual(max((light + .05) / .05, 1.05 / (light + .05)), 3)

    def test_tie_is_class_zero_and_no_change_has_no_ring(self):
        probabilities = [[.5, .5]] * 169
        context = [{'x': 0, 'y': 0}]
        queries = [{'x': i % 13, 'y': i // 13} for i in range(169)]
        record = {'context': context, 'queries': queries, 'models': {'test': {'conditions': [
            {'probabilities': probabilities, 'flip_index': None},
            {'probabilities': probabilities, 'flip_index': 0},
        ]}}}
        self.assertEqual(overlays(record)['test'][1]['classes'], [0] * 169)
        self.assertEqual(overlays(record)['test'][1]['changed_cells'], [])
        self.assertIsNone(overlays(record)['test'][1]['ring'])

    def test_readable_summary_and_current_evidence(self):
        self.assertIn('Median / maximum across all 16 flips', self.page)
        self.assertEqual(self.page.count('<th scope="row">'), 3)
        self.assertNotIn('Max |ΔP|', self.page)
        self.assertNotIn('independent verification pending', self.page)
        self.assertIn('Artifact checked by a separate internal reviewer in the same organization. Not independently reproduced.', self.page)


if __name__ == '__main__':
    unittest.main()
