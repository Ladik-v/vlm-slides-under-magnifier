import argparse
import ast
import json
import os
from collections import Counter

PAGE_KEYS = ['image_{}', 'page_{}', 'slide_{}']


def pages_of(row):
    ev = row.get('evidence_pages')
    if isinstance(ev, str):
        try:
            ev = ast.literal_eval(ev)
        except (ValueError, SyntaxError):
            ev = [int(t) for t in ev.replace('[', ' ').replace(']', ' ').replace(',', ' ').split()]
    if isinstance(ev, int):
        ev = [ev]
    return [int(p) for p in ev or []]


def page_image(row, p):
    for k in PAGE_KEYS:
        if row.get(k.format(p)) is not None:
            return row[k.format(p)]
    for k in ['images', 'slides', 'pages']:
        if isinstance(row.get(k), list) and len(row[k]) >= p:
            return row[k][p - 1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--split', default='test')
    ap.add_argument('--pool', type=int, default=400)
    ap.add_argument('--max-scan', type=int, default=6000)
    ap.add_argument('--per-deck', type=int, default=2)
    ap.add_argument('--max-answer-words', type=int, default=5)
    ap.add_argument('--repo', default='NTT-hil-insight/SlideVQA')
    args = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(os.path.join(args.out, 'slides'), exist_ok=True)
    pool_path = os.path.join(args.out, 'pool.jsonl')
    ds = load_dataset(args.repo, split=args.split, streaming=True)
    kept, per_deck, scanned, why = [], Counter(), 0, Counter()
    for row in ds:
        if scanned == 0:
            print('columns:', {k: type(v).__name__ for k, v in row.items()})
            print('first row meta:', {k: (v if not hasattr(v, 'size') else 'image') for k, v in row.items() if k.startswith(('qa', 'question', 'answer', 'evidence', 'reason', 'deck'))})
        scanned += 1
        if scanned > args.max_scan or len(kept) >= args.pool:
            break
        ev = pages_of(row)
        deck = row.get('deck_name', '')
        ans = str(row.get('answer', '')).strip()
        rt = str(row.get('reasoning_type', '')).lower()
        at = str(row.get('answer_type', '')).lower()
        if len(ev) != 1:
            why['not_single_page'] += 1
            continue
        if rt and 'single' not in rt:
            why['not_single_hop'] += 1
            continue
        if not ans or len(ans.split()) > args.max_answer_words or 'multi' in at:
            why['answer_filter'] += 1
            continue
        if per_deck[deck] >= args.per_deck:
            why['deck_cap'] += 1
            continue
        img = page_image(row, ev[0])
        if img is None:
            why['no_image'] += 1
            continue
        qa_id = str(row.get('qa_id', f'row{scanned}'))
        rel = os.path.join('slides', f'{qa_id}.png')
        img.convert('RGB').save(os.path.join(args.out, rel))
        per_deck[deck] += 1
        kept.append({'id': qa_id, 'question': row['question'], 'answers': [ans], 'image': rel, 'answer_type': at, 'reasoning_type': rt, 'deck': deck, 'page': ev[0]})
    with open(pool_path, 'w') as f:
        for it in kept:
            f.write(json.dumps(it) + '\n')
    print(f'scanned {scanned}, kept {len(kept)}, rejected {dict(why)}')
    print('answer types:', dict(Counter(it['answer_type'] for it in kept)))
    print('pool written to', pool_path)


if __name__ == '__main__':
    main()
