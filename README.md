# Slides under a magnifying glass: a VLM with crop and zoom

Modern Methods and Algorithms of Generative Artificial Intelligence course project. A vision-language model, Qwen3-VL-4B-Instruct, answers questions about presentation slides that need a small detail: a label, a number on a chart, a link between blocks. We give it a crop/zoom tool, let it pick regions, and measure whether zooming helps compared with looking at the whole slide.

Main result: on 120 SlideVQA test questions zoom did not help. A single view of the slide at its native resolution (768 visual tokens) scores 0.79 exact match (EM) at about 800 tokens per question. The zoom agent with the same first look scores 0.71 at about 2660 tokens. SlideVQA slides are only 1024 px on the long side, so a downscaled overview loses little and zoom has little to recover.

On our own 2560 px slides (40 questions, many with small print) the picture changes: the zoom agent reaches 0.85 against 0.62 for the same overview without zoom. A bigger single view without zoom (768 tokens, 0.90) or the 1500-token view (0.93) is still as good, within noise. Zoom helps when the first look is too small to read, and it adds wrong-box errors, so the net gain is zero once the first look is large enough.

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

## Results on own and dense slides (2560 px, exact match)

| Mode | Own, 20 questions | Dense, 20 questions | All 40 | Tokens / question (all 40) |
|---|---|---|---|---|
| A, 256 tokens | 0.70 | 0.55 | 0.62 | about 310 |
| A768, 768 tokens | 1.00 | 0.80 | 0.90 | about 830 |
| B, 1500 tokens | 0.95 | 0.90 | 0.93 | about 1570 |
| C, overview plus 4 tiles | 0.75 | 0.95 | 0.85 | about 3900 |
| D, zoom agent | 0.90 | 0.80 | 0.85 | about 1500 to 1800 |

The own questions cover four detail types, 5 each (axis label, value on a chart, small text, relation between blocks). The dense set has 20 questions on 8 slides with tables and bar charts in small print. With 20 to 40 questions an interval is about plus or minus 15 to 20 points. D beats A by 22.5 points on all 40 (13 questions against 4, p = 0.049) and A768 beats A by 27.5 points (13 against 2, p = 0.007). D against A768 and B is inside the noise. Fixed tiles (C) are best on the dense slides (0.95) and cost about 3900 tokens.

## Error analysis (SlideVQA test)

Each of A and D makes 41 errors on the 120 test questions. We sorted 30 sampled errors per mode:

| Category | A | D |
|---|---|---|
| `no_zoom` (text too small, no zoom requested) | 12 | 1 |
| `wrong_region` (box misses or cuts the evidence) | 0 | 10 |
| `format` (right meaning, EM is 0) | 8 | 10 |
| `reasoning` | 5 | 6 |
| `misread` (right region, wrong text) | 3 | 2 |
| `bad_gold` | 2 | 1 |

Zoom removes most of the `no_zoom` errors and adds wrong-box errors, which is why the totals are equal. About a third of the errors of D are format mismatches, so EM understates both modes. The categories were assigned in one pass by an AI assistant and checked by one author, and the shares come from samples of 30. The tables are in `errors/`.

## Demo examples

Each picture shows the slide with the numbered box that mode D chose (left) and the crop the model received (right). More cases with exact numbers are in `examples/expected_results.md`.

**Zoom works.** Question on `dense_12`: "What value is shown for DreamZero (Scratch) on task 2, Remove Hat from Mannequin?" Mode A answers 28.6, D boxes the dark bar of task 2, reads 85.7 in the crop and is right.

![dense_12: the box on the tallest bar and the crop with the value 85.7](demo_outputs/replay_dense_12.png)

**Zoom reads a small diagram.** SlideVQA test question 32: "What is the abbreviation for Sales Qualified Leads?" A answers SAL, D boxes the tiny diagram block and reads SQL.

![SlideVQA 32: the box on the small funnel diagram and the enlarged block Sales Qualified Leads (SQL)](demo_outputs/replay_32.png)

**Axis label.** Own question 05: "What is the label of the vertical axis of the plot?" A invents "Decoding cost per token (in $)", D boxes the rotated title and reads "FLOPs per generated token, one layer".

![own_05: the box on the vertical axis title and its crop](demo_outputs/replay_own_05.png)

**Right region, wrong bar.** Dense question 14: "What value is shown for DreamZero (Scratch) on PnP Easy?" Gold 93.8. D boxes the right group and the crop shows 93.8, but the model answers 2.1, the value of another bar. The zoom gave it the detail and it used the wrong bar.

![dense_14: the box on the PnP Easy group and the crop with the bars 2.1, 17.6, 52.1 and 93.8](demo_outputs/replay_dense_14.png)

**The box cuts the evidence.** SlideVQA test question 877: "How many players were in the pre-1983 market?" Gold 5. A answers 5, D boxes the list but the box ends in the middle of the last name, and D answers 4.

![SlideVQA 877: the box on the Players list stops before the last line](demo_outputs/replay_877.png)

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

## Limitations

- SlideVQA slides are 1024 px on the long side, so an overview of 768 tokens shows them without any loss and zoom has nothing to recover. This explains most of the test results.
- 120 test questions, one run, greedy decoding. Differences of a few points are within noise.
- The 40 own and dense questions are ours, on 17 slides, and the dense slides were chosen for small print, which favours any method that reads small print.
- The model is a 4-bit quantised 4B model on a 16 GB Mac. A larger model or a higher precision can change the order of the modes.
- Modes A to C use a text answer prompt and D, E use JSON. Mode F (A with the JSON prompt) shows no format effect.
- The answer match is strict (normalised exact match), so some correct answers in other wording count as errors. About a third of the errors of D are of this kind.
- The error categories come from samples of 30 errors per mode, assigned in one pass by an AI assistant and checked by one author.

## Related work

DeepEyes (reinforcement learning for thinking with images, the next step beyond this project), GAZE (zoom agent framework, medical images), FocusVTC (a crop model for documents based on Qwen3.5).

## Data and licence note

Slide images and questions come from SlideVQA (Hugging Face `NTT-hil-insight/SlideVQA`, gated). `data/pool.jsonl`, `results/` and the montages in `errors/` contain material derived from it, so keep this repository private and follow the dataset terms.
