# Contributing

Fourteen guidelines for the code in this repo, and how each one is checked. They follow Google's
Python Style Guide, "Good Enough Practices in Scientific Computing" (Wilson et al., 2017), the Good
Research Code Handbook and the ML Code Completeness Checklist.

## Before you push

```bash
pip install -r requirements.txt ruff==0.17.5 pytest
ruff format .               # rewrites the files in the agreed style
ruff check .                # reports what breaks a rule
pytest -q                   # 28 checks, a few seconds, no GPU
python bouncing_shapes.py   # the dataset's preview
python plot_runs.py         # collapse.png, rebuilt from logs/
```

The same steps run on every push (`.github/workflows/check.yml`), with the format step only
reporting. The settings are in `pyproject.toml`.

## The guidelines

| # | guideline | checked by |
|---|---|---|
| 1 | One job per function, about 40 lines at most | ruff: 40 statements, 10 branches, 6 arguments |
| 2 | A choice between named things is a table, not an `if` chain (`LABELS` in `plot_runs.py`) | review |
| 3 | Nothing runs at import: every script has a `main()` | review |
| 4 | Each piece of logic lives in one place | review |
| 5 | A file opens with what it does, how to run it, and who wrote it | ruff: a docstring is required |
| 6 | Comments say why, never what | review |
| 7 | Every rule behind a published number has a test | review; the tests run on every push |
| 8 | A formatter and a linter decide style | ruff, 100 characters |
| 9 | Computing is kept apart from reading and writing files | review |
| 10 | No state shared between functions behind their backs | ruff flags `global`; review |
| 11 | Numbers, columns and paths have names | review |
| 12 | Every figure in the README's table is in a log in `logs/`; dependencies are written down | `tests/test_plot_runs.py` compares the table with the last row of each log; `baselines.py` measures the two baselines; `requirements.txt` |
| 13 | Type hints on every function; a docstring with `Args:` and `Returns:` where the names do not say it | ruff |
| 14 | One folder per role, one level deep; only the commands a reader runs sit in the root | review |

"Review" means a person reads the change against the guideline: no tool checks it.

## Where a file goes

| folder | role | goes here |
|---|---|---|
| root | what a reader runs or imports | `train_jepa.py`, `plot_runs.py`, `baselines.py`, `bouncing_shapes.py`, `jepa.py`, and the files that describe the repo |
| `logs/` | what came out | the printed output of one training run per file, never edited |
| `tests/` | how the code is checked | one file per module tested |

## Changing something that produces a number

The table in the README is the last row of each file in `logs/`. Those rows depend on
`bouncing_shapes.py` (the clips), `jepa.py` (the model and the loss) and `train_jepa.py` (the
training step, the probe and the seed). A full run takes about 9 minutes on an Apple M4 Pro, so the
tests do not train. A change to any of the three files must show that the published numbers still
follow:

- `pytest -q` must pass: it checks the probe's split (6,348 rows to fit, 1,588 held out), the
  dataset's shapes and that an index always gives the same clip, the collapse metric, and that the
  README's table equals the logs.
- One epoch of each variant must print the first row of its log. For example
  `python train_jepa.py --epochs 1` must print `1   0.1731    7.268      2.40`, the first row of
  `logs/run.txt`. This holds on Apple MPS with the versions in `requirements.txt`; on another
  device the last digits may differ.
- `python plot_runs.py` must draw the same lines: it reads only `logs/`.

A change that is meant to move the numbers is a new experiment: it gets a new log file and a new row
in `LABELS`, and the four published logs stay as they are.

## Not covered

No workflow trains the model or runs `baselines.py`: the check on every push has no GPU and stops
at the tests, the dataset preview and the plot.
