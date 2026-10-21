import json
import math
from PIL import Image, ImageDraw, ImageFont

TOKEN_PX = 32 * 32
PALETTE = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#008080']


def fit(image, tokens, upscale=False):
    w, h = image.size
    target = tokens * TOKEN_PX
    if w * h <= target and not upscale:
        k = 1.0
    else:
        k = math.sqrt(target / (w * h))
    nw, nh = max(32, round(w * k / 32) * 32), max(32, round(h * k / 32) * 32)
    if (nw, nh) == (w, h):
        return image
    return image.resize((nw, nh), Image.LANCZOS)


def visual_tokens(image):
    return round(image.width * image.height / TOKEN_PX)


def zoom(image, box, pad=0.03, target=1280):
    w, h = image.size
    x1, y1, x2, y2 = (v / 1000 for v in box)
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(1, x2 + pad), min(1, y2 + pad)
    area = (x2 - x1) * (y2 - y1)
    if x2 <= x1 or y2 <= y1 or area < 0.01 or area > 0.7:
        return None
    crop = image.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
    k = target / max(crop.size)
    return crop.resize((round(crop.width * k), round(crop.height * k)), Image.LANCZOS)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return round(inter / union, 4) if union > 0 else 0.0


def tiles(image, grid=2, overlap=0.1, target=1280):
    w, h = image.size
    out = []
    for r in range(grid):
        for c in range(grid):
            x1, x2 = max(0, c / grid - overlap / 2), min(1, (c + 1) / grid + overlap / 2)
            y1, y2 = max(0, r / grid - overlap / 2), min(1, (r + 1) / grid + overlap / 2)
            crop = image.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            k = target / max(crop.size)
            out.append(crop.resize((round(crop.width * k), round(crop.height * k)), Image.LANCZOS))
    return out


def _json_objects(text):
    start = text.find('{')
    while start != -1:
        depth, in_str, esc = 0, False, False
        end = None
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is not None:
            try:
                yield json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        start = text.find('{', start + 1)


def _as_action(obj):
    if not isinstance(obj, dict):
        return None
    action = obj.get('action')
    if action == 'answer' and obj.get('answer') is not None:
        return {'action': 'answer', 'answer': str(obj['answer']).strip()}
    if action == 'zoom':
        box = obj.get('box')
        if isinstance(box, (list, tuple)) and len(box) == 4:
            try:
                x1, y1, x2, y2 = (float(v) for v in box)
            except (TypeError, ValueError):
                return None
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            return {'action': 'zoom', 'box': [round(max(0.0, min(1000.0, v)), 1) for v in (x1, y1, x2, y2)]}
    return None


def parse_action(text):
    for obj in _json_objects(text or ''):
        act = _as_action(obj)
        if act:
            return act
    return None


def draw_boxes(image, boxes, width=4):
    img = image.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    size = max(14, w // 40)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for i, b in enumerate(boxes):
        col = PALETTE[i % len(PALETTE)]
        x1, y1, x2, y2 = b[0] / 1000 * w, b[1] / 1000 * h, b[2] / 1000 * w, b[3] / 1000 * h
        d.rectangle([x1, y1, x2, y2], outline=col, width=width)
        d.rectangle([x1, y1, x1 + size, y1 + size * 1.2], fill=col)
        d.text((x1 + 3, y1), str(i + 1), fill='white', font=font)
    return img
