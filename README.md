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
| `main.ipynb` | main notebook with all outputs: setup, data, backend test, runs, results, own questions, error analysis, demo, limitations |
| `backend.py` | one interface `generate(messages, images, max_tokens)` with MLX and transformers backends |
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
| `own_slides/questions.csv` | the 20 own questions (5 per detail type) |
| `own_slides/questions_dense.csv` | 20 extra questions on 8 dense slides (small text, tables, bar charts) |
| `tests/test_all.py` | 24 unit tests |
| `examples/` | demo examples for reproduction: 6 slides, 9 questions, recorded answers, launcher script (see below) |
| `demo_outputs/` | montages of five replayed records (slide with boxes and crops) |
| `docs/` | the presentation (pptx) |

## Setup

Python 3.10 or newer (3.11 or 3.12 recommended). The real model needs an Apple Silicon Mac; 16 GB of RAM is enough for the 4-bit model.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python prepare_data.py --split test --pool 400
python3 -I tests/test_all.py
jupyter notebook main.ipynb
```

- `hf auth login` asks for a Hugging Face read token. SlideVQA is a gated dataset: accept its terms on the dataset page first.
- `prepare_data.py` rewrites `data/pool.jsonl`. Keep `splits.json` as it is. If a regenerated pool misses an id from `splits.json`, `data.get_split` raises a `KeyError`; do not regenerate splits, restore the original `data/pool.jsonl` instead.
- In the notebook pick the kernel of the `.venv` (register it with `python -m ipykernel install --user --name magnifier --display-name "Python (magnifier)"`).
- `BACKEND_KIND` in the setup cell switches between `'mlx'` (the Mac) and `'transformers'` (Kaggle or Colab).
- Every long run appends one line per finished question to a jsonl file and skips finished ids on restart. Any run cell can be stopped and started again.
- The notebook already contains outputs of the full run. Running all cells again loads the model and skips every finished run.

## Reproducing the demo

The folder `examples/` holds everything needed to repeat the demo from the talk. It has 6 slides (`examples/slides/`), 9 questions with gold answers (`examples/questions.csv`), the recorded answers of modes A, B, C and D (`examples/replay.jsonl`), a table of the expected results (`examples/expected_results.md`) and three montages (`examples/previews/`: slide with the numbered box on the left, the crop the model received on the right).

Setup is the same as above, without the data download:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Replay, no model needed (any machine, a minute): `python examples/run_demo.py`, then open the address it prints (http://127.0.0.1:7860). The Replay tab lists the 9 recorded runs of mode D. Pick one to see the slide with the box, the crop, the step log and the answer. The Live tab needs the model.
2. Live, with the model (Apple Silicon Mac, 16 GB, the 4-bit model of about 6 GB downloads on first run): `python examples/run_demo.py --live`. In the Live tab pick a slide, keep or edit the question, choose mode D and press Run. For a comparison run the same question with mode A. On a machine without a Mac use `--backend transformers` (a GPU with 16 GB or Colab T4, needs `pip install torch transformers accelerate`).
3. Check against the recorded answers, no browser: `python examples/run_demo.py --check` runs all 9 questions in modes A and D and prints exact match now and as recorded, the answer and the boxes. It takes about 2 minutes on an M4, plus the model load.

What to expect, from `examples/expected_results.md`:

| Question (slide) | A, no zoom | D, zoom agent | Box of D |
|---|---|---|---|
| value of DreamZero (Scratch) on task 2 (`d_vw_p83`) | 28.6, wrong | 85.7, right (gold 85.7) | the tallest bar of task 2 |
| vertical axis label of the top chart (`d_vw_p83`) | Average Task Success Rate (%), wrong | Average Task Progress (%), right | the rotated axis title |
| vocabulary size of Llama 2 (`d_l2_p33`) | 3K, wrong | 32k, right | the table row of Llama 2 |
| model size of Llama 2 (`d_l2_p33`) | right | wrong: another row | a different row of the table |
| FVD of DIAMOND-HQ on CS:GO (`d_vw_p58`) | 17.5, wrong | 368.5, right | the row label, not the number |

The other four questions are in the table file. Greedy decoding makes the output nearly deterministic, but another machine, backend or library version can change a character in an answer or shift a box by a few units, so compare the pattern (A wrong and D right on 8 of 9, D wrong on `dense_08`, a misplaced box on `dense_17`), not the exact text. The examples were picked after looking at the results, so 8 of 9 for D is not an accuracy estimate; the accuracy on 120 unselected questions is in the results table above.

The slides are figures from course material and papers, included in small number only for reproduction. Keep the repository private (see the licence note at the end).

## Method in short

- Data: SlideVQA test split, questions with one evidence slide, answers of at most five words, at most two questions per deck. The model sees only the evidence slide, never the whole deck. Questions are not selected by model errors. 400 in the pool, 30 dev and 120 test in `splits.json`.
- Agent protocol: JSON in the prompt, not native tool calling. The model replies with `{"action": "zoom", "box": [x1, y1, x2, y2]}` (coordinates 0 to 1000 relative to the full slide) or `{"action": "answer", "answer": "..."}`. A zoom crops the original slide and upsamples the crop to 1280 px.
- Loop guards: at most 3 zooms; a box with IoU above 0.7 against an earlier box ends the zooming; two bad regions end it too; one retry after a JSON parse error, then an empty answer logged as a format error.
- Image budget: a visual token covers about 32x32 px. Low budget 256 tokens, high budget 1500.
- Metrics: exact match and token F1 after normalisation (lowercase, thousands commas removed, punctuation dropped, articles dropped). An empty or invalid answer scores 0 and stays in the denominator.

## What is left to do

1. Own and dense questions. The 20 own questions and the 20 dense-slide questions are written (`own_slides/questions.csv`, `own_slides/questions_dense.csv`). Run sections 6.1 to 6.3 of the notebook: A at 768 tokens on the own questions, modes A, A768, B, C, D on the dense ones, then the comparison tables. Slide images are not committed (see `.gitignore`); pass the `own_slides/` folder to the other team member separately.
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
