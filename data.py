import csv
import json
import os
import random
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_synthetic(root, n=12, seed=0, size=(1280, 720)):
    os.makedirs(os.path.join(root, 'slides'), exist_ok=True)
    rng = random.Random(seed)
    names = ['Sales', 'Costs', 'Profit', 'Users', 'Churn', 'Growth']
    items = []
    for i in range(n):
        im = Image.new('RGB', size, 'white')
        d = ImageDraw.Draw(im)
        d.text((40, 20), f'Quarterly report {i + 1}', fill='black', font=_font(36))
        vals = [rng.randint(10, 99) for _ in names]
        for j, (nm, v) in enumerate(zip(names, vals)):
            x = 120 + j * 170
            top = 660 - v * 5
            d.rectangle([x, top, x + 100, 660], fill=(60, 90, 200))
            d.text((x + 25, top - 18), str(v), fill='black', font=_font(14))
            d.text((x + 10, 670), nm, fill='black', font=_font(16))
        k = rng.randrange(len(names))
        rel = os.path.join('slides', f'syn_{i:03d}.png')
        im.save(os.path.join(root, rel))
        items.append({'id': f'syn_{i:03d}', 'question': f'In quarterly report {i + 1}, what is the value of the {names[k]} bar?', 'answers': [str(vals[k])], 'image': rel, 'answer_type': 'number', 'deck': 'synthetic'})
    with open(os.path.join(root, 'pool.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    return items
