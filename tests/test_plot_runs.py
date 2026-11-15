"""Tests for plot_runs.py and for the README's table: both must follow from the logs in logs/.

Code written by Claude Code on Nov 15, 2026, for the author.
"""

from pathlib import Path

from plot_runs import LABELS, LOGS, parse_log, read_runs

README = Path(__file__).parent.parent / "README.md"
EPOCHS = 12  # every published run


def test_parse_log_keeps_epoch_rows_and_skips_everything_else():
    log = [
        "device: mps | ema=True var_reg=True",
        "epoch     loss  enc_std probe_err  (px)",
        "    1   0.1731    7.268      2.40  (px)",
        "    2   0.0601    3.913      2.36  (px)",
        "",
        "collapse: enc_std -> ~0, loss -> 0.0000",
    ]
    assert parse_log(log) == ([1, 2], [7.268, 3.913], [2.40, 2.36])


def test_parse_log_ignores_a_row_out_of_sequence():
    assert parse_log(["    1 0.1 7.0 2.4 (px)", "    3 0.1 6.0 2.2 (px)"]) == ([1], [7.0], [2.4])


def test_every_log_is_labelled_and_has_12_epochs():
    assert sorted(p.name for p in LOGS.glob("*.txt")) == sorted(LABELS)
    runs = read_runs()
    assert list(runs) == list(LABELS.values())  # plotted in the order of the README's table
    for epochs, enc_std, probe_err in runs.values():
        assert epochs == list(range(1, EPOCHS + 1))
        assert len(enc_std) == len(probe_err) == EPOCHS


def test_readme_table_is_the_last_row_of_each_log():
    table = {}
    for line in README.read_text().splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] in LABELS.values():
            table[cells[0]] = (cells[2], cells[3], cells[4])
    assert len(table) == len(LABELS)
    for name, label in LABELS.items():
        last = (LOGS / name).read_text().splitlines()[1 + EPOCHS].split()
        assert last[0] == str(EPOCHS)
        assert table[label] == (last[1], last[2], f"{last[3]} px"), label
