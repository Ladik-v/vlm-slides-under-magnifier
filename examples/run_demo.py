import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import agent
import demo
import metrics
from PIL import Image


def load_items():
    with open(os.path.join(HERE, 'questions.csv'), newline='') as f:
        return [{'id': r['id'], 'image': r['image'], 'question': r['question'], 'answers': [a.strip() for a in r['answer'].split(';')]} for r in csv.DictReader(f)]


def make_backend(kind):
    from backend import make_backend as mk
    ids = {'mlx': 'mlx-community/Qwen3-VL-4B-Instruct-4bit', 'transformers': 'Qwen/Qwen3-VL-4B-Instruct'}
    b = mk(kind, model_id=ids[kind])
    agent.set_backend(b)
    return b


def check(items, backend, modes):
    recorded = {(r['id'], r['mode']): r for r in agent.read_jsonl(os.path.join(HERE, 'replay.jsonl'))}
    print(f'{"id":10} {"mode":4} {"now":>6} {"recorded":>9}  answer')
    for it in items:
        image = Image.open(os.path.join(HERE, it['image'])).convert('RGB')
        for m in modes:
            rec = agent.run(image, it['question'], m, backend)
            now = metrics.em(rec['answer'], it['answers'])
            old = metrics.em(recorded[(it['id'], m)]['answer'], it['answers'])
            boxes = [[round(v) for v in s['box']] for s in rec['steps'] if s['valid']]
            print(f'{it["id"]:10} {m:4} {now:6.0f} {old:9.0f}  {rec["answer"]!r} {boxes}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--live', action='store_true')
    p.add_argument('--check', action='store_true')
    p.add_argument('--backend', default='mlx', choices=['mlx', 'transformers'])
    p.add_argument('--port', type=int, default=7860)
    a = p.parse_args()
    items = load_items()
    backend = make_backend(a.backend) if a.live or a.check else None
    if a.check:
        check(items, backend, 'AD')
        return
    app = demo.build_app(items, HERE, backend=backend, replay_path=os.path.join(HERE, 'replay.jsonl'))
    app.launch(server_port=a.port)


if __name__ == '__main__':
    main()
