import json
import os
import time
from PIL import Image
from tools import fit, zoom, iou, tiles, parse_action

CFG = {'low': 256, 'high': 1500, 'target': 1280, 'max_tokens': 160, 'max_zooms': 3, 'iou_repeat': 0.7}
BACKEND = None

SYS_AGENT = '''You see a presentation slide, possibly downscaled. Answer the question.
If the needed text or numbers are too small to read reliably, request a zoom.
Reply with exactly one JSON object:
{"action": "zoom", "box": [x1, y1, x2, y2]}  (coordinates 0 to 1000 relative to the full slide)
or {"action": "answer", "answer": "<short phrase or number>"}'''
SYS_ONE = SYS_AGENT + '\nYou must request exactly one zoom first, then answer.'
SYS_JSON = 'You see a presentation slide, possibly downscaled. Answer the question.\nReply with exactly one JSON object: {"action": "answer", "answer": "<short phrase or number>"}'
SYS_PLAIN = 'You see a presentation slide. Answer the question with only a short phrase or a number, nothing else.'
FORMAT_REMINDER = 'Invalid reply. Reply with exactly one JSON object: {"action": "zoom", "box": [x1, y1, x2, y2]} or {"action": "answer", "answer": "..."}.'
ANSWER_NOW = 'Answer now with {"action": "answer", "answer": "..."}.'


def set_backend(b):
    global BACKEND
    BACKEND = b


def _img():
    return {'type': 'image'}


def _txt(t):
    return {'type': 'text', 'text': t}


def _msg(role, *parts):
    return {'role': role, 'content': list(parts)}


def _gen(b, msgs, images, rec):
    b.last_tokens = 0
    out = b.generate(msgs, images, CFG['max_tokens'])
    rec['tokens'] += int(b.last_tokens)
    return out


def clean(text):
    act = parse_action(text)
    if act and act['action'] == 'answer':
        return act['answer']
    lines = [l.strip() for l in (text or '').strip().splitlines() if l.strip()]
    if not lines:
        return ''
    return lines[0].strip('"\'').rstrip('.').strip()


def _single(b, images, intro, question, rec, system=SYS_PLAIN):
    msgs = [_msg('system', _txt(system)), _msg('user', *[_img() for _ in images], _txt(f'{intro}Question: {question}'))]
    rec['answer'] = clean(_gen(b, msgs, images, rec))


def mode_a(b, image, question, rec):
    _single(b, [fit(image, CFG['low'])], '', question, rec)


def mode_b(b, image, question, rec):
    _single(b, [fit(image, CFG['high'], upscale=True)], '', question, rec)


def mode_c(b, image, question, rec):
    imgs = [fit(image, CFG['low'])] + [fit(t, CFG['high']) for t in tiles(image, target=CFG['target'])]
    intro = 'Image 1 is the full slide, downscaled. Images 2 to 5 are enlarged crops of its top-left, top-right, bottom-left and bottom-right parts (they overlap slightly). '
    _single(b, imgs, intro, question, rec)


def _agent(b, image, question, rec, system, max_zoom, require_zoom):
    images = [fit(image, CFG['low'])]
    msgs = [_msg('system', _txt(system)), _msg('user', _img(), _txt(f'Question: {question}'))]
    boxes, zooms, bad, retried, nudged, must_answer = [], 0, 0, False, False, False
    for _ in range(10):
        raw = _gen(b, msgs, images, rec)
        act = parse_action(raw)
        msgs.append(_msg('assistant', _txt(json.dumps(act) if act else raw)))
        if act is None:
            rec['steps'].append({'box': None, 'raw': raw, 'valid': False, 'reason': 'parse'})
            if retried:
                rec['error'], rec['answer'] = 'format', ''
                return
            retried = True
            msgs.append(_msg('user', _txt(FORMAT_REMINDER)))
            continue
        if act['action'] == 'answer':
            if require_zoom and zooms == 0 and not nudged and not must_answer:
                nudged = True
                rec['steps'].append({'box': None, 'raw': raw, 'valid': False, 'reason': 'answer_before_zoom'})
                msgs.append(_msg('user', _txt('Request one zoom first, then answer.')))
                continue
            rec['answer'] = act['answer']
            return
        box = act['box']
        if must_answer:
            rec['steps'].append({'box': box, 'raw': raw, 'valid': False, 'reason': 'limit'})
            if retried:
                rec['error'], rec['answer'] = 'format', ''
                return
            retried = True
            msgs.append(_msg('user', _txt(ANSWER_NOW)))
            continue
        if any(iou(box, p) > CFG['iou_repeat'] for p in boxes):
            rec['steps'].append({'box': box, 'raw': raw, 'valid': False, 'reason': 'repeat'})
            must_answer = True
            msgs.append(_msg('user', _txt('You already inspected that region. ' + ANSWER_NOW)))
            continue
        crop = zoom(image, box, target=CFG['target'])
        if crop is None:
            bad += 1
            rec['steps'].append({'box': box, 'raw': raw, 'valid': False, 'reason': 'bad_region'})
            if bad >= 2:
                must_answer = True
                msgs.append(_msg('user', _txt('Bad region. ' + ANSWER_NOW)))
            else:
                msgs.append(_msg('user', _txt('Bad region. Choose another region or answer.')))
            continue
        zooms += 1
        boxes.append(box)
        rec['steps'].append({'box': box, 'raw': raw, 'valid': True, 'reason': None})
        images.append(fit(crop, CFG['high']))
        if zooms >= max_zoom:
            must_answer = True
            text = 'Enlarged region. This was the last zoom. ' + ANSWER_NOW
        else:
            text = 'Enlarged region (box coordinates stay relative to the full slide). Reply with one JSON object.'
        msgs.append(_msg('user', _img(), _txt(text)))
    rec['error'], rec['answer'] = 'format', ''


def mode_d(b, image, question, rec):
    _agent(b, image, question, rec, SYS_AGENT, CFG['max_zooms'], False)


def mode_e(b, image, question, rec):
    _agent(b, image, question, rec, SYS_ONE, 1, True)


def mode_f(b, image, question, rec):
    _single(b, [fit(image, CFG['low'])], '', question, rec, SYS_JSON)


MODES = {'F': mode_f, 'A': mode_a, 'B': mode_b, 'C': mode_c, 'D': mode_d, 'E': mode_e}


def run(image, question, mode, backend=None):
    b = backend or BACKEND
    rec = {'id': None, 'mode': mode, 'question': question, 'gold': None, 'steps': [], 'answer': '', 'tokens': 0, 'seconds': 0.0, 'error': None}
    t0 = time.time()
    MODES[mode](b, image.convert('RGB'), question, rec)
    rec['seconds'] = round(time.time() - t0, 2)
    return rec


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_dataset(items, mode, out_path, backend=None, root='.', log_every=10):
    done = {r['id'] for r in read_jsonl(out_path) if r['mode'] == mode}
    todo = [it for it in items if it['id'] not in done]
    for i, it in enumerate(todo):
        image = Image.open(os.path.join(root, it['image'])).convert('RGB')
        rec = run(image, it['question'], mode, backend)
        rec['id'], rec['gold'] = it['id'], it['answers']
        with open(out_path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if (i + 1) % log_every == 0 or i + 1 == len(todo):
            print(f'mode {mode}: {len(done) + i + 1}/{len(items)}')
    return [r for r in read_jsonl(out_path) if r['mode'] == mode]


def replay_crops(image, rec):
    out = []
    for s in rec['steps']:
        if s['valid']:
            out.append(zoom(image, s['box'], target=CFG['target']))
    return out
