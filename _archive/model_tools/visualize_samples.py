from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

root_candidates = [Path().resolve(), Path().resolve().parent]
DATA_DIR = None
for root in root_candidates:
    candidate = root / "data" / "processed" / "windows"
    if candidate.exists():
        DATA_DIR = candidate
        break

if DATA_DIR is None:
    raise FileNotFoundError("Could not locate data/processed/windows directory")

labels_path = DATA_DIR / "labels.csv"
if not labels_path.exists():
    raise FileNotFoundError(
        "labels.csv not found. Run scripts/generate_windows.py to create windows."
    )

labels = pd.read_csv(labels_path)
if labels.empty:
    raise ValueError("No window metadata available in labels.csv")

n_samples = min(10, len(labels))
sample = labels.sample(n=n_samples, random_state=42).reset_index(drop=True)

rows = int(np.ceil(n_samples / 5)) or 1
cols = min(5, n_samples) or 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3))
axes = np.array(axes).reshape(rows, cols)

for ax in axes.flatten():
    ax.axis("off")

last_im = None
for ax, (_, row) in zip(axes.flatten(), sample.iterrows()):
    window = np.load(DATA_DIR / row["window_file"])
    last_im = ax.imshow(window, aspect="auto", origin="lower", cmap="viridis")
    title = f"label={row.get('label', 'NA')} | {row['window_file']}"
    ax.set_title(title, fontsize=9)
    ax.axis("off")

if last_im is not None:
    cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_label("Amplitude (z-score)")

fig.suptitle("Random CSI Windows", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()


