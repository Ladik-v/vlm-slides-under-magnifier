import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image

import agent
import data
import metrics
import tools
from helpers import ScriptedBackend, make_slides


def scripted(replies):
    it = iter(replies)
    return ScriptedBackend(lambda m, i: next(it))


def img(w=1280, h=720):
    return Image.new('RGB', (w, h), 'white')


class TestMetrics(unittest.TestCase):
    def test_norm(self):
        self.assertEqual(metrics.norm('The 1,234.5 Apples!'), ['1234.5', 'apples'])
        self.assertEqual(metrics.norm('3.'), ['3'])
        self.assertEqual(metrics.norm(''), [])

    def test_em_f1(self):
        self.assertEqual(metrics.em('The Cat', ['cat']), 1.0)
        self.assertEqual(metrics.em('', ['']), 0.0)
        self.assertEqual(metrics.em('dog', ['cat', 'dog']), 1.0)
        self.assertEqual(metrics.f1('big red cat', ['red cat']), 0.8)
        self.assertEqual(metrics.f1('', ['cat']), 0.0)
        self.assertEqual(metrics.f1('x x', ['x']), 0.6667)

    def test_num_ok(self):
        self.assertEqual(metrics.num_ok('1,020', ['1000']), 1.0)
        self.assertEqual(metrics.num_ok('1100', ['1000']), 0.0)
        self.assertIsNone(metrics.num_ok('none', ['1000']))

    def test_ci_and_mcnemar(self):
        m, (lo, hi) = metrics.ci([1, 0, 1, 1])
        self.assertEqual(m, 0.75)
        self.assertLessEqual(lo, m)
        self.assertGreaterEqual(hi, m)
        self.assertEqual(metrics.mcnemar([1, 0], [1, 0]), (0, 0, 1.0))
        x, y, p = metrics.mcnemar([1] * 10 + [0] * 2, [0] * 10 + [1] * 2)
        self.assertEqual((x, y), (10, 2))
        self.assertLess(p, 0.05)

    def test_score_empty_counts(self):
        recs = [{'id': 'a', 'gold': ['1'], 'answer': '1', 'steps': [{'valid': True}], 'tokens': 10, 'seconds': 1},
                {'id': 'b', 'gold': ['2'], 'answer': '', 'steps': [], 'tokens': 20, 'seconds': 3, 'error': 'format'}]
        s = metrics.score(recs)
        self.assertEqual(s['em'], 0.5)
        self.assertEqual(s['zoom_rate'], 0.5)
        self.assertEqual(s['format_errors'], 0.5)
        self.assertEqual(s['tokens'], 15.0)


class TestTools(unittest.TestCase):
    def test_zoom_ok_and_bad(self):
        c = tools.zoom(img(), [100, 100, 500, 500])
        self.assertEqual(max(c.size), 1280)
        self.assertIsNone(tools.zoom(img(), [0, 0, 1000, 1000]))
        self.assertIsNone(tools.zoom(img(), [100, 100, 110, 110], pad=0))
        self.assertIsNone(tools.zoom(img(), [500, 500, 100, 100], pad=0))

    def test_iou(self):
        self.assertEqual(tools.iou([0, 0, 100, 100], [0, 0, 100, 100]), 1.0)
        self.assertEqual(tools.iou([0, 0, 100, 100], [200, 200, 300, 300]), 0.0)
        self.assertEqual(tools.iou([0, 0, 100, 100], [50, 0, 150, 100]), 0.3333)

    def test_parse_action(self):
        self.assertEqual(tools.parse_action('{"action": "answer", "answer": 42}'), {'action': 'answer', 'answer': '42'})
        self.assertEqual(tools.parse_action('Sure! {"action": "zoom", "box": [10, 20, 300, 400]} ok'), {'action': 'zoom', 'box': [10.0, 20.0, 300.0, 400.0]})
        self.assertEqual(tools.parse_action('{"action":"zoom","box":[300,400,10,20]}')['box'], [10.0, 20.0, 300.0, 400.0])
        self.assertEqual(tools.parse_action('{"action":"zoom","box":[-5,0,2000,10]}')['box'], [0.0, 0.0, 1000.0, 10.0])
        self.assertIsNone(tools.parse_action('no json here'))
        self.assertIsNone(tools.parse_action('{"action": "zoom", "box": [1, 2]}'))
        self.assertIsNone(tools.parse_action('{"action": "answer"}'))
        self.assertIsNone(tools.parse_action('{"action": "zoom", "box": '))
        self.assertEqual(tools.parse_action('{"x": {"y": 1}} {"action": "answer", "answer": "a}b"}')['answer'], 'a}b')

    def test_fit_and_tiles(self):
        low = tools.fit(img(2000, 1000), 400)
        self.assertLessEqual(tools.visual_tokens(low), 420)
        self.assertEqual(low.size[0] % 32, 0)
        self.assertEqual(tools.fit(img(320, 320), 400).size, (320, 320))
        up = tools.fit(img(640, 360), 1500, upscale=True)
        self.assertGreater(tools.visual_tokens(up), 1300)
        ts = tools.tiles(img())
        self.assertEqual(len(ts), 4)
        self.assertTrue(all(max(t.size) == 1280 for t in ts))

    def test_draw_boxes(self):
        out = tools.draw_boxes(img(), [[100, 100, 400, 400], [500, 500, 900, 900]])
        self.assertEqual(out.size, (1280, 720))


class TestAgent(unittest.TestCase):
    def run_mode(self, mode, replies):
        b = scripted(replies)
        return agent.run(img(), 'q?', mode, backend=b), b

    def test_single_modes(self):
        for m in 'ABC':
            rec, b = self.run_mode(m, ['42.'])
            self.assertEqual(rec['answer'], '42')
            self.assertEqual(rec['steps'], [])
        self.assertEqual(b.calls[0]['n_images'], 5)

    def test_f_json_answer(self):
        rec, b = self.run_mode('F', ['{"action": "answer", "answer": "17"}'])
        self.assertEqual(rec['answer'], '17')
        self.assertEqual(b.calls[0]['n_images'], 1)
        rec, b = self.run_mode('F', ['17.'])
        self.assertEqual(rec['answer'], '17')

    def test_d_zoom_then_answer(self):
        rec, b = self.run_mode('D', ['{"action":"zoom","box":[0,0,400,400]}', '{"action":"answer","answer":"7"}'])
        self.assertEqual(rec['answer'], '7')
        self.assertEqual([s['valid'] for s in rec['steps']], [True])
        self.assertEqual([c['n_images'] for c in b.calls], [1, 2])
        self.assertGreater(rec['tokens'], 0)

    def test_d_limit_three_zooms(self):
        z = lambda x: json.dumps({'action': 'zoom', 'box': [x, 0, x + 300, 300]})
        rec, b = self.run_mode('D', [z(0), z(350), z(700), '{"action":"answer","answer":"x"}'])
        self.assertEqual(rec['answer'], 'x')
        self.assertEqual(sum(s['valid'] for s in rec['steps']), 3)

    def test_d_zoom_after_limit_then_answer(self):
        z = lambda x: json.dumps({'action': 'zoom', 'box': [x, 0, x + 300, 300]})
        rec, b = self.run_mode('D', [z(0), z(350), z(700), z(0), '{"action":"answer","answer":"y"}'])
        self.assertEqual(rec['answer'], 'y')
        self.assertEqual(rec['steps'][-1]['reason'], 'limit')

    def test_d_zoom_after_limit_twice_is_error(self):
        z = lambda x: json.dumps({'action': 'zoom', 'box': [x, 0, x + 300, 300]})
        rec, b = self.run_mode('D', [z(0), z(350), z(700), z(0), z(0)])
        self.assertEqual(rec['error'], 'format')
        self.assertEqual(rec['answer'], '')

    def test_d_repeat_box_forces_answer(self):
        zb = '{"action":"zoom","box":[0,0,400,400]}'
        rec, b = self.run_mode('D', [zb, zb, '{"action":"answer","answer":"z"}'])
        self.assertEqual(rec['answer'], 'z')
        self.assertEqual([s['reason'] for s in rec['steps']], [None, 'repeat'])

    def test_d_bad_region_retry_message(self):
        rec, b = self.run_mode('D', ['{"action":"zoom","box":[0,0,1000,1000]}', '{"action":"answer","answer":"w"}'])
        self.assertEqual(rec['steps'][0]['reason'], 'bad_region')
        self.assertEqual(rec['answer'], 'w')

    def test_d_parse_retry_once(self):
        rec, b = self.run_mode('D', ['hmm', '{"action":"answer","answer":"ok"}'])
        self.assertEqual(rec['answer'], 'ok')
        self.assertIsNone(rec['error'])
        rec, b = self.run_mode('D', ['hmm', 'still no'])
        self.assertEqual(rec['error'], 'format')
        self.assertEqual(rec['answer'], '')

    def test_history_keeps_only_first_action(self):
        seen = []
        replies = iter(['{"action":"zoom","box":[0,0,400,400]}\n\n{"action":"answer","answer":"fake"}', '{"action":"answer","answer":"real"}'])

        def policy(messages, images):
            seen.append(list(messages))
            return next(replies)

        rec = agent.run(img(), 'q?', 'D', backend=ScriptedBackend(policy))
        self.assertEqual(rec['answer'], 'real')
        assistant = [m['content'][0]['text'] for m in seen[1] if m['role'] == 'assistant']
        self.assertEqual(len(assistant), 1)
        self.assertNotIn('fake', assistant[0])

    def test_e_requires_zoom(self):
        rec, b = self.run_mode('E', ['{"action":"answer","answer":"1"}', '{"action":"zoom","box":[0,0,400,400]}', '{"action":"answer","answer":"2"}'])
        self.assertEqual(rec['answer'], '2')
        self.assertEqual([s['reason'] for s in rec['steps']], ['answer_before_zoom', None])
        rec, b = self.run_mode('E', ['{"action":"answer","answer":"1"}', '{"action":"answer","answer":"3"}'])
        self.assertEqual(rec['answer'], '3')


class TestResume(unittest.TestCase):
    def test_run_dataset_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = make_slides(tmp, n=4)
            out = os.path.join(tmp, 'r.jsonl')
            b1 = ScriptedBackend(lambda m, i: '1')
            agent.run_dataset(items[:2], 'A', out, backend=b1, root=tmp)
            self.assertEqual(len(b1.calls), 2)
            b2 = ScriptedBackend(lambda m, i: '1')
            recs = agent.run_dataset(items, 'A', out, backend=b2, root=tmp)
            self.assertEqual(len(b2.calls), 2)
            self.assertEqual(len(recs), 4)
            self.assertEqual(len({r['id'] for r in recs}), 4)
            agent.run_dataset(items, 'B', out, backend=b2, root=tmp)
            self.assertEqual(len(agent.read_jsonl(out)), 8)


class TestData(unittest.TestCase):
    def test_splits_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = make_slides(tmp, n=20)
            p = os.path.join(tmp, 'splits.json')
            s1 = data.make_splits(items, p, n_dev=5, n_test=10)
            s2 = data.make_splits(items[::-1], p, n_dev=7, n_test=3)
            self.assertEqual(s1, s2)
            self.assertFalse(set(s1['dev']) & set(s1['test']))
            self.assertEqual((len(s1['dev']), len(s1['test'])), (5, 10))
            self.assertEqual(len(data.get_split(items, s1, 'dev')), 5)

    def test_size_stats_and_own_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = make_slides(tmp, n=3)
            st = data.size_stats(items, tmp)
            self.assertEqual(st['median'], 1280)
            self.assertFalse(st['zoom_has_headroom'])
            p = data.make_own_template(os.path.join(tmp, 'own', 'questions.csv'))
            self.assertEqual(data.load_own(p), [])
            with open(p) as f:
                self.assertEqual(len(f.read().strip().splitlines()), 21)


if __name__ == '__main__':
    unittest.main()
