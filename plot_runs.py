"""Plot enc_std and probe_err per epoch, one line per run, from the logs in logs/.

    python plot_runs.py   # reads logs/*.txt, writes collapse.png; works from any folder

Who wrote what: the author wrote the log parser; the matplotlib calls were written with an
assistant. The labels, the paths and main() were added by Claude Code on Nov 15, 2026; the values
plotted are unchanged.
"""

from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
LOGS = ROOT / "logs"
PLOT = ROOT / "collapse.png"

# Log file -> the variant's name in the README's table. A log that is not listed here is labelled
# with its file name.
LABELS = {
    "run--no-ema_--no-var-reg.txt": "1: neither",
    "run--no-var-reg.txt": "2: EMA only",
    "run.txt": "3: EMA + variance term",
    "run_ema_normalize.txt": "4: EMA + normalize",
}
# Columns of an epoch row, as train_jepa.py prints it: epoch, loss, enc_std, probe_err.
ENC_STD_COLUMN, PROBE_ERR_COLUMN = 2, 3
# Where the dashed line is drawn: the mean predictor's error as train_jepa.py prints it. The README
# gives the measured range (baselines.py).
MEAN_PREDICTOR_ERR_PX = 5.62

Series = tuple[list[int], list[float], list[float]]  # epochs, enc_std, probe_err


def parse_log(lines: list[str]) -> Series:
    """Read the epoch rows of one training log.

    A row counts when its first field is the next epoch number, so the header and the closing
    lines are skipped.
    """
    epochs, enc_std, probe_err = [], [], []
    i = 1
    for line in lines:
        fields = line.split()
        if fields and fields[0].isnumeric() and int(fields[0]) == i:
            i += 1
            epochs.append(int(fields[0]))
            enc_std.append(float(fields[ENC_STD_COLUMN]))
            probe_err.append(float(fields[PROBE_ERR_COLUMN]))
    return epochs, enc_std, probe_err


def read_runs(logs: Path = LOGS) -> dict[str, Series]:
    """Parse every log in `logs`, in file-name order. Returns label -> series."""
    runs = {}
    for f in sorted(logs.glob("*.txt")):
        runs[LABELS.get(f.name, f.stem)] = parse_log(f.read_text().splitlines())
    return runs


def plot(runs: dict[str, Series], out: Path = PLOT) -> None:
    """Draw the two panels and save them to `out`."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for label, (epochs, enc_std, probe_err) in runs.items():
        ax1.plot(epochs, enc_std, marker="o", label=label)
        ax2.plot(epochs, probe_err, marker="o", label=label)

    ax1.set_yscale("log")  # enc_std spans 0.001 to 7: on a normal axis three runs sit flat on 0
    ax1.set(title="enc_std (spread of embeddings)", xlabel="epoch")
    ax2.axhline(MEAN_PREDICTOR_ERR_PX, ls="--", color="gray", label="floor: always guess the mean")
    ax2.set(title="probe_err (px, lower is better)", xlabel="epoch")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    """Rebuild collapse.png from the logs."""
    plot(read_runs())
    print(f"wrote {PLOT}")


if __name__ == "__main__":
    main()
