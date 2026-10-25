import json
import os
import random
from PIL import Image, ImageDraw, ImageFont
from backend import Backend
from tools import visual_tokens


class ScriptedBackend(Backend):
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def generate(self, messages, images, max_tokens=256):
        text = self.policy(messages, images)
        self.calls.append({'n_images': len(images), 'reply': text})
        self.last_tokens = sum(visual_tokens(im) for im in images) + sum(len(p.get('text', '')) // 4 for m in messages for p in m['content']) + len(text) // 4
        return text


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_slides(root, n=12, seed=0, size=(1280, 720)):
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
        rel = os.path.join('slides', f'slide_{i:03d}.png')
        im.save(os.path.join(root, rel))
        items.append({'id': f'slide_{i:03d}', 'question': f'In quarterly report {i + 1}, what is the value of the {names[k]} bar?', 'answers': [str(vals[k])], 'image': rel, 'answer_type': 'number', 'deck': 'test'})
    with open(os.path.join(root, 'pool.jsonl'), 'w') as f:
        for it in items:
            f.write(json.dumps(it) + '\n')
    return items
