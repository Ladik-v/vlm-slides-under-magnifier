import os
import tempfile
from tools import visual_tokens


class Backend:
    last_tokens = 0

    def generate(self, messages, images, max_tokens=256):
        raise NotImplementedError


class MLXBackend(Backend):
    def __init__(self, model_id='mlx-community/Qwen3-VL-4B-Instruct-4bit'):
        from mlx_vlm import load
        self.model_id = model_id
        self.model, self.processor = load(model_id)

    def generate(self, messages, images, max_tokens=256):
        from mlx_vlm import generate
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, im in enumerate(images):
                p = os.path.join(tmp, f'{i}.png')
                im.convert('RGB').save(p)
                paths.append(p)
            out = generate(self.model, self.processor, prompt, image=paths or None, max_tokens=max_tokens, temperature=0.0, verbose=False)
        self.last_tokens = int(getattr(out, 'prompt_tokens', 0) + getattr(out, 'generation_tokens', 0))
        return getattr(out, 'text', out)


class TransformersBackend(Backend):
    def __init__(self, model_id='Qwen/Qwen3-VL-4B-Instruct'):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.model_id = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype='auto', device_map='auto')
        self.processor = AutoProcessor.from_pretrained(model_id)

    def generate(self, messages, images, max_tokens=256):
        it = iter(images)
        filled = []
        for m in messages:
            parts = [{'type': 'image', 'image': next(it).convert('RGB')} if p['type'] == 'image' else p for p in m['content']]
            filled.append({'role': m['role'], 'content': parts})
        inputs = self.processor.apply_chat_template(filled, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt').to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen = out[:, inputs['input_ids'].shape[1]:]
        self.last_tokens = int(out.shape[1])
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


class MockBackend(Backend):
    def __init__(self, policy):
        self.policy = policy
        self.calls = []

    def generate(self, messages, images, max_tokens=256):
        text = self.policy(messages, images)
        self.calls.append({'n_images': len(images), 'reply': text})
        self.last_tokens = sum(visual_tokens(im) for im in images) + sum(len(p.get('text', '')) // 4 for m in messages for p in m['content']) + len(text) // 4
        return text


def make_backend(kind='mlx', **kw):
    return {'mlx': MLXBackend, 'transformers': TransformersBackend}[kind](**kw)
