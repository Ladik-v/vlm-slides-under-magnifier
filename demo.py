import os
from PIL import Image
import agent
from tools import draw_boxes


def step_log(rec):
    lines = []
    for i, s in enumerate(rec['steps'], 1):
        tag = 'ok' if s['valid'] else f"skipped ({s.get('reason')})"
        lines.append(f'{i}. box={s["box"]} {tag}\n   raw: {s["raw"][:200].replace(chr(10), " ")}')
    lines.append(f'answer: {rec["answer"]!r}' + (f'  [error: {rec["error"]}]' if rec.get('error') else ''))
    if rec.get('gold') is not None:
        lines.append(f'gold: {rec["gold"]}')
    lines.append(f'tokens: {rec["tokens"]}, seconds: {rec["seconds"]}')
    return '\n'.join(lines)


def explain(image, rec):
    boxes = [s['box'] for s in rec['steps'] if s['valid']]
    shown = draw_boxes(image, boxes) if boxes else image.convert('RGB')
    crops = agent.replay_crops(image, rec)
    return shown, [(c, f'zoom {i + 1}') for i, c in enumerate(crops)], step_log(rec)


def montage(image, rec, path, height=360):
    shown, gallery, _ = explain(image, rec)
    parts = [shown] + [c for c, _ in gallery]
    parts = [p.resize((round(p.width * height / p.height), height)) for p in parts]
    canvas = Image.new('RGB', (sum(p.width for p in parts) + 8 * (len(parts) - 1), height), 'white')
    x = 0
    for p in parts:
        canvas.paste(p, (x, 0))
        x += p.width + 8
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    canvas.save(path)
    return path


def live(image, question, mode, backend=None):
    rec = agent.run(image, question, mode, backend)
    return explain(image, rec) + (rec['answer'],)


def replay_choices(path, mode='D', limit=10):
    recs = [r for r in agent.read_jsonl(path) if r['mode'] == mode]
    return recs[:limit]


def build_app(items, root, backend=None, replay_path=None, replay_mode='D'):
    import gradio as gr
    by_id = {it['id']: it for it in items}
    replay = {r['id']: r for r in replay_choices(replay_path, replay_mode)} if replay_path else {}

    def load(sid):
        it = by_id[sid]
        return Image.open(os.path.join(root, it['image'])).convert('RGB'), it['question']

    def on_pick(sid):
        im, q = load(sid)
        return im, q

    def on_live(sid, upload, question, mode):
        im = upload if upload is not None else load(sid)[0]
        shown, gallery, log, ans = live(im, question, mode, backend)
        return shown, gallery, log, ans

    def on_replay(rid):
        rec = replay[rid]
        im = load(rid)[0]
        shown, gallery, log = explain(im, rec)
        return shown, gallery, log, rec['answer']

    with gr.Blocks() as app:
        with gr.Tab('Live'):
            sid = gr.Dropdown(list(by_id), label='Slide', value=next(iter(by_id)))
            upload = gr.Image(type='pil', label='Or upload a slide')
            q = gr.Textbox(label='Question')
            mode = gr.Radio(list('ABCDE'), value='D', label='Mode')
            go = gr.Button('Run')
            out_img = gr.Image(label='Slide with boxes')
            gallery = gr.Gallery(label='Zooms')
            log = gr.Textbox(label='Steps', lines=8)
            ans = gr.Textbox(label='Answer')
            sid.change(on_pick, sid, [upload, q])
            go.click(on_live, [sid, upload, q, mode], [out_img, gallery, log, ans])
        if replay:
            with gr.Tab('Replay'):
                rid = gr.Dropdown(list(replay), label='Precomputed record', value=next(iter(replay)))
                r_img = gr.Image(label='Slide with boxes')
                r_gal = gr.Gallery(label='Zooms')
                r_log = gr.Textbox(label='Steps', lines=8)
                r_ans = gr.Textbox(label='Answer')
                rid.change(on_replay, rid, [r_img, r_gal, r_log, r_ans])
    return app
