import csv
import json
import os
import random
from collections import defaultdict
import numpy as np
from PIL import Image

DETAIL_TYPES = ['axis_label', 'chart_value', 'small_text', 'block_relation']
OWN_COLUMNS = ['id', 'image', 'question', 'answer', 'detail_type']


def read_pool(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def size_stats(items, root='.'):
    sides = []
    for it in items:
        with Image.open(os.path.join(root, it['image'])) as im:
            sides.append(max(im.size))
    p10, med, p90 = (int(v) for v in np.percentile(sides, [10, 50, 90]))
    out = {'n': len(sides), 'p10': p10, 'median': med, 'p90': p90}
    out['zoom_has_headroom'] = med >= 1400
    return out


def answer_kind(it):
    if it.get('answer_type'):
        return it['answer_type']
    return 'number' if any(c.isdigit() for c in it['answers'][0]) else 'text'


def make_splits(items, path='splits.json', n_dev=30, n_test=120, seed=0):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if len(items) < n_dev + n_test:
        raise ValueError(f'pool has {len(items)} items, need {n_dev + n_test}')
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[answer_kind(it)].append(it['id'])
    for g in groups.values():
        rng.shuffle(g)
    order = []
    while any(groups.values()):
        for k in sorted(groups):
            if groups[k]:
                order.append(groups[k].pop())
    chosen = order[:n_dev + n_test]
    rng.shuffle(chosen)
    splits = {'dev': chosen[:n_dev], 'test': chosen[n_dev:], 'seed': seed}
    with open(path, 'w') as f:
        json.dump(splits, f, indent=1)
    return splits


def get_split(items, splits, name):
    by_id = {it['id']: it for it in items}
    return [by_id[i] for i in splits[name]]


def make_own_template(path='own_slides/questions.csv', per_type=5):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(OWN_COLUMNS)
        n = 0
        for t in DETAIL_TYPES:
            for _ in range(per_type):
                n += 1
                w.writerow([f'own_{n:02d}', '', '', '', t])
    return path


def load_own(path='own_slides/questions.csv'):
    out = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if not (r['image'] and r['question'] and r['answer']):
                continue
            out.append({'id': r['id'], 'question': r['question'], 'answers': [a.strip() for a in r['answer'].split(';') if a.strip()], 'image': os.path.basename(r['image']), 'answer_type': r['detail_type'], 'detail_type': r['detail_type']})
    return out
