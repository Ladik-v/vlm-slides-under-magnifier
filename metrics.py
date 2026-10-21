import re
import numpy as np
from collections import Counter
from scipy.stats import binomtest


def norm(s):
    s = re.sub(r'(?<=\d),(?=\d)', '', str(s).lower())
    s = re.sub(r'[^\w.\s]', ' ', s)
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return [t.strip('.') for t in s.split() if t.strip('.')]


def as_list(golds):
    return [golds] if isinstance(golds, str) else list(golds)


def em(pred, golds):
    p = norm(pred)
    if not p:
        return 0.0
    return float(any(p == norm(g) for g in as_list(golds)))


def f1(pred, golds):
    p = norm(pred)
    best = 0.0
    for g in map(norm, as_list(golds)):
        common = sum((Counter(p) & Counter(g)).values())
        if common:
            pr, rc = common / len(p), common / len(g)
            best = max(best, 2 * pr * rc / (pr + rc))
    return round(best, 4)


def to_number(s):
    m = re.search(r'-?\d+(?:\.\d+)?', re.sub(r'(?<=\d),(?=\d)', '', str(s)))
    return float(m.group()) if m else None


def num_ok(pred, golds, tol=0.05):
    p = to_number(pred)
    if p is None:
        return None
    for g in as_list(golds):
        v = to_number(g)
        if v is None:
            continue
        if v == 0:
            if p == 0:
                return 1.0
        elif abs(p - v) / abs(v) <= tol:
            return 1.0
    return 0.0


def ci(scores, n=2000, seed=0):
    s = np.asarray(scores, float)
    rng = np.random.default_rng(seed)
    means = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(n)]
    return round(float(s.mean()), 4), np.percentile(means, [2.5, 97.5]).round(4).tolist()


def paired_ci(a, b, n=2000, seed=0):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return round(float(d.mean()), 4), np.percentile(means, [2.5, 97.5]).round(4).tolist()


def mcnemar(a, b):
    x = sum(1 for u, v in zip(a, b) if u and not v)
    y = sum(1 for u, v in zip(a, b) if v and not u)
    if x + y == 0:
        return x, y, 1.0
    return x, y, round(binomtest(x, x + y, 0.5).pvalue, 4)


def valid_zooms(rec):
    return sum(1 for s in rec.get('steps', []) if s.get('valid'))


def per_item(records):
    return {r['id']: (em(r.get('answer') or '', r['gold']), f1(r.get('answer') or '', r['gold'])) for r in records}


def score(records):
    n = len(records)
    ems = [em(r.get('answer') or '', r['gold']) for r in records]
    f1s = [f1(r.get('answer') or '', r['gold']) for r in records]
    nums = [x for x in (num_ok(r.get('answer') or '', r['gold']) for r in records) if x is not None]
    em_m, em_ci = ci(ems)
    f1_m, f1_ci = ci(f1s)
    vz = [valid_zooms(r) for r in records]
    return {
        'n': n,
        'em': em_m,
        'em_ci': em_ci,
        'f1': f1_m,
        'f1_ci': f1_ci,
        'zoom_rate': round(sum(1 for v in vz if v > 0) / n, 4),
        'mean_zooms': round(sum(vz) / n, 2),
        'format_errors': round(sum(1 for r in records if r.get('error') == 'format') / n, 4),
        'tokens': round(sum(r.get('tokens', 0) for r in records) / n, 1),
        'seconds': round(sum(r.get('seconds', 0) for r in records) / n, 2),
        'num_5pct': round(sum(nums) / len(nums), 4) if nums else None,
        'n_numeric': len(nums),
    }


def compare(rec_a, rec_b):
    da, db = {r['id']: r for r in rec_a}, {r['id']: r for r in rec_b}
    ids = sorted(set(da) & set(db))
    a = [em(da[i].get('answer') or '', da[i]['gold']) for i in ids]
    b = [em(db[i].get('answer') or '', db[i]['gold']) for i in ids]
    fa = [f1(da[i].get('answer') or '', da[i]['gold']) for i in ids]
    fb = [f1(db[i].get('answer') or '', db[i]['gold']) for i in ids]
    only_a, only_b, p = mcnemar(a, b)
    d_em, d_em_ci = paired_ci(a, b)
    d_f1, d_f1_ci = paired_ci(fa, fb)
    return {'n': len(ids), 'd_em': d_em, 'd_em_ci': d_em_ci, 'd_f1': d_f1, 'd_f1_ci': d_f1_ci, 'only_a': only_a, 'only_b': only_b, 'mcnemar_p': p}
