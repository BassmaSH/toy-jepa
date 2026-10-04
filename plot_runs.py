"""Plot enc_std and probe_err per epoch, one line per run, from the logs in logs/."""
from pathlib import Path

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for f in sorted(Path("logs").glob("*.txt")):
    lines = open(f)
    epochs, enc_std, probe_err = [],[],[]
    i=1
    for line in lines:
        
        l = line.split()
        if l and l[0].isnumeric() and int(l[0]) == i:
            i+=1
            epochs.append(int(l[0]))
            enc_std.append(float(l[2]))
            probe_err.append(float(l[3]))
    ax1.plot(epochs, enc_std, marker="o", label=f.stem)
    ax2.plot(epochs, probe_err, marker="o", label=f.stem)

ax1.set_yscale("log")   # enc_std spans 0.001 to 7: on a normal axis three runs sit flat on 0
ax1.set(title="enc_std (spread of embeddings)", xlabel="epoch")
ax2.axhline(5.62, ls="--", color="gray", label="floor: always guess the mean")
ax2.set(title="probe_err (px, lower is better)", xlabel="epoch")
ax2.legend()
fig.tight_layout()
fig.savefig("collapse.png", dpi=150)