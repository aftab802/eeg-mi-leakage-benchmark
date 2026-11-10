"""Four-class CNN (left fist, right fist, both feet, rest) on PhysioNet EEGMMIDB.

Builds the 64x64 representation (64 channels x 64 window-averaged time points)
for all 109 subjects, then trains the six-block CNN. --protocol trial is the
original 90/10 trial-level split; --protocol L0 splits by subject.
Run with --epochs 50 and --epochs 100 for the two configurations in the paper.
"""
import numpy as np
from _common import parser, save, subjects_arg

from cybex_eeg.mi_experiments import run_fourclass
from cybex_eeg.physionet import load_fourclass_subject, to_64x64

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--subjects", default="1-109")
    p.add_argument("--protocol", default="trial", choices=["trial", "L0"])
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    trials, labels, subs = [], [], []
    for s in subjects_arg(a.subjects):
        t, l = load_fourclass_subject(s, data_path=a.data_path, rng=rng)
        trials.append(t); labels.append(l); subs.append(s)
        print(f"Subject {s:03d} loaded -> {t.shape}")
    trials = np.stack(trials)
    X = to_64x64(trials)
    y = np.stack(labels).reshape(-1, 4).argmax(1)
    g = np.repeat(subs, trials.shape[1])
    print("X:", X.shape, "class counts:", np.bincount(y))
    save(run_fourclass(X, y, g, protocol=a.protocol, epochs=a.epochs, seed=a.seed),
         f"fourclass_{a.protocol}_{a.epochs}ep", a.outdir)
