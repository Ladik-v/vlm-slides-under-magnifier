# Slides under a magnifying glass: a VLM with crop and zoom

Course project. A vision-language model, Qwen3-VL-4B-Instruct, answers questions about presentation slides that need a small detail: a label, a number on a chart, a link between blocks. We give it a crop/zoom tool, let it pick regions, and measure whether zooming helps compared with looking at the whole slide.

Main result on 120 SlideVQA test questions: zoom did not help. A single view of the slide at its native resolution (768 visual tokens) scores 0.79 exact match at about 800 tokens per question. The zoom agent with the same first look scores 0.71 at about 2660 tokens. SlideVQA slides are only 1024 px on the long side, so a downscaled overview loses little and zoom has little to recover. The own-slides experiment (section 6 of the notebook) is meant to test slides with a higher resolution.

## Results (test, n = 120, greedy decoding, Qwen3-VL-4B-Instruct 4-bit via mlx-vlm)

| Mode | What the model sees | EM (95% CI) | F1 | Zoomed on | Tokens / question | Seconds / question |
|---|---|---|---|---|---|---|
| A | whole slide, 256 visual tokens | 0.66 (0.57 to 0.74) | 0.73 | 0% | 309 | 1.6 |
| A768 | whole slide, 768 visual tokens | 0.79 (0.72 to 0.86) | 0.87 | 0% | 799 | 4.6 |
| B | whole slide, 1500 visual tokens | 0.82 (0.75 to 0.88) | 0.86 | 0% | 1584 | 10.9 |
| C | 256-token overview plus 4 tiles (2x2, 10% overlap) | 0.75 (0.67 to 0.83) | 0.83 | 0% | 5003 | 36.6 |
| D | agent: 256-token overview, up to 3 zooms | 0.66 (0.57 to 0.74) | 0.74 | 84% | 1770 | 12.8 |
| D768 | agent with a 768-token overview | 0.71 (0.63 to 0.78) | 0.80 | 87% | 2664 | 18.5 |
| E | agent forced to zoom exactly once | 0.63 (0.55 to 0.72) | 0.74 | 99% | 1737 | 11.7 |
| F | A with a JSON answer prompt (format control) | 0.66 (0.57 to 0.74) | 0.72 | 0% | 336 | 2.1 |

With 120 questions an accuracy near 0.7 has an interval of about plus or minus 9 points, so small gaps are not effects. The notebook (section 5) gives paired bootstrap intervals and exact McNemar tests. The clear results are A (256) below A768 and B, and D768 below A768 (12 questions against 2, p = 0.013). B against A768 is inside the noise (p = 0.375).

## Repository layout

| File | Content |
|---|---|
| `slides_under_magnifier_v4.ipynb` | main notebook with all outputs: setup, data, backend test, runs, results, own questions, error analysis, demo, limitations |
| `backend.py` | one interface `generate(messages, images, max_tokens)` with MLX, transformers and mock backends |
| `tools.py` | zoom, IoU, tiles, image resizing to a token budget, JSON action parser, box drawing |
| `agent.py` | modes A to F, loop guards, resumable `run_dataset` |
| `data.py` | pool loading, size statistics, frozen splits, own-questions template and loader |
| `metrics.py` | normalisation, EM, F1, bootstrap intervals, paired comparison, McNemar |
| `demo.py` | Gradio app with a live tab and a replay tab |
| `prepare_data.py` | streams SlideVQA from Hugging Face and builds `data/pool.jsonl` and the slide images |
| `splits.json` | 30 dev and 120 test question ids. Frozen, never change it |
| `data/pool.jsonl` | the 400 selected questions (slide images are not in the repo) |
| `results/*.jsonl` | one line per answered question for every run |
| `errors/` | tables to label by hand and the image montages they refer to |
| `own_slides/questions.csv` | template for the 15 own questions |
| `tests/test_all.py` | 24 unit tests |

## Setup

Python 3.10 or newer (3.11 or 3.12 recommended). The real model needs an Apple Silicon Mac; 16 GB of RAM is enough for the 4-bit model.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python prepare_data.py --split test --pool 400
python3 -I tests/test_all.py
jupyter notebook slides_under_magnifier_v4.ipynb
```

- `hf auth login` asks for a Hugging Face read token. SlideVQA is a gated dataset: accept its terms on the dataset page first.
- `prepare_data.py` rewrites `data/pool.jsonl`. Keep `splits.json` as it is. If a regenerated pool misses an id from `splits.json`, `data.get_split` raises a `KeyError`; do not regenerate splits, restore the original `data/pool.jsonl` instead.
- In the notebook pick the kernel of the `.venv` (register it with `python -m ipykernel install --user --name magnifier --display-name "Python (magnifier)"`).
- `BACKEND_KIND` in the setup cell switches between `'mlx'` (real model) and `'mock'` (scripted replies that test the pipeline, no model needed). Numbers from the mock backend mean nothing.
- Every long run appends one line per finished question to a jsonl file and skips finished ids on restart. Any run cell can be stopped and started again.
- The notebook already contains outputs of the full run. Running all cells again loads the model and skips every finished run.

## Method in short

- Data: SlideVQA test split, questions with one evidence slide, answers of at most five words, at most two questions per deck. The model sees only the evidence slide, never the whole deck. Questions are not selected by model errors. 400 in the pool, 30 dev and 120 test in `splits.json`.
- Agent protocol: JSON in the prompt, not native tool calling. The model replies with `{"action": "zoom", "box": [x1, y1, x2, y2]}` (coordinates 0 to 1000 relative to the full slide) or `{"action": "answer", "answer": "..."}`. A zoom crops the original slide and upsamples the crop to 1280 px.
- Loop guards: at most 3 zooms; a box with IoU above 0.7 against an earlier box ends the zooming; two bad regions end it too; one retry after a JSON parse error, then an empty answer logged as a format error.
- Image budget: a visual token covers about 32x32 px. Low budget 256 tokens, high budget 1500.
- Metrics: exact match and token F1 after normalisation (lowercase, thousands commas removed, punctuation dropped, articles dropped). An empty or invalid answer scores 0 and stays in the denominator.

## What is left to do

1. Own questions. Put 8 to 10 slides in `own_slides/` (export them at about 2500 px on the long side, because zoom can only help when the slide is larger than the model's budget) and fill `own_slides/questions.csv`: 15 questions, 5 per detail type (`axis_label`, `chart_value`, `small_text`, `block_relation`), at most 2 to 3 per slide. Several accepted answers go in one cell separated by `;`. Choose questions by the detail they need, not by where the model fails. Then run section 6 of the notebook. Slide images are not committed (see `.gitignore`).
2. Error analysis. Fill the `category` column in `errors/mlx_error_table.csv` (60 errors of modes D and A), `errors/mlx_discordant.csv` (16 questions where A and D disagree) and `errors/mlx_discordant_FD.csv` (14 questions where F and D disagree). The notebook never overwrites these files. Categories:
   - `no_zoom`: the model did not request a zoom although the needed text was small, and answered wrongly
   - `wrong_region`: the box is not on the evidence
   - `misread`: the region is right but the text was read wrongly
   - `format`: the meaning is right but exact match is 0 (count this share separately, it is a metric error)
   - `reasoning`: the question needs a calculation, a comparison or several slides
   - `bad_gold`: the gold label itself looks wrong (optional extra category)

   Agree on the rules together on about 10 shared rows first, then split the rest between two people.
3. Demo check. Run section 8, open the Gradio address printed in the cell, run one question in the Live tab and open one record in the Replay tab. Save two or three screenshots for the report.
4. Report: method, results table, the accuracy against cost figure (section 5.5), error analysis, limitations, related work.
5. Freeze the code 4 hours before the deadline. Do not change `splits.json`, `data/pool.jsonl` or `results/`.

## Limitations

- SlideVQA slides are 1024 px on the long side, so an overview of 768 tokens shows them without any loss and zoom has nothing to recover. This explains most of the results above.
- 120 test questions, one run, greedy decoding. Differences of a few points are within noise.
- The model is a 4-bit quantised 4B model on a 16 GB Mac. A larger model or a higher precision can change the order of the modes.
- The answer match is strict (normalised exact match), so some correct answers in other wording count as errors. The share is measured in the error analysis.

## Related work

DeepEyes (reinforcement learning for thinking with images, the next step beyond this project), GAZE (zoom agent framework, medical images), FocusVTC (a crop model for documents based on Qwen3.5).

## Data and licence note

Slide images and questions come from SlideVQA (Hugging Face `NTT-hil-insight/SlideVQA`, gated). `data/pool.jsonl`, `results/` and the montages in `errors/` contain material derived from it, so keep this repository private and follow the dataset terms.
