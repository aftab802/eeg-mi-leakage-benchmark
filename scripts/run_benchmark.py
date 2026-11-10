"""Leakage-safe motor-imagery benchmark and matched-protocol ablation.

Runs the four architectures with their own data configurations under the
requested protocol and, with --ablation, the 1D CNN under L0 and L2 with
everything else held constant. Results go to one JSON file.
"""
import numpy as np
from _common import parser, save

from cybex_eeg import mi_experiments as E
from cybex_eeg.physionet import (EXECUTED_LR_RUNS, IMAGERY_LR_RUNS, IRREGULAR_SUBJECTS, load_fourclass_subject,
                                 load_lr_dataset, to_64x64)

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--models", default="csp,eegnet,conv1d,fourclass")
    p.add_argument("--ablation", action="store_true", help="1D CNN under L0 and L2")
    p.add_argument("--fourclass-protocol", default="trial", choices=["trial", "L0"])
    p.add_argument("--max-subjects", type=int, default=None, help="use only the first N subjects (quick check)")
    p.add_argument("--epochs", type=int, default=None, help="override every model's epoch budget (quick check)")
    a = p.parse_args()
    cap = (lambda r: list(r)[:a.max_subjects]) if a.max_subjects else list
    ep = {} if a.epochs is None else {"epochs": a.epochs}
    models, out = a.models.split(","), {}
    if "eegnet" in models:
        X, y, g = load_lr_dataset(cap(s for s in range(1, 71) if s not in IRREGULAR_SUBJECTS), data_path=a.data_path)
        out["eegnet_L0"] = E.run_eegnet(X, y, g, "L0", seed=a.seed, **ep)
    if "csp" in models:
        X, y, g = load_lr_dataset(cap(range(1, 110)), resample=160.0, data_path=a.data_path)
        out["csp_cnn_L0"] = E.run_csp_cnn(X, y, g, "L0", seed=a.seed, **ep)
    if "conv1d" in models or a.ablation:
        X, y, g = load_lr_dataset(cap(range(1, 110)), runs=EXECUTED_LR_RUNS + IMAGERY_LR_RUNS, tmax=2.0 - 1 / 160,
                                  resample=160.0, data_path=a.data_path)
        out["conv1d_L0"] = E.run_conv1d(X, y, g, "L0", seed=a.seed, **ep)
        if a.ablation:
            out["conv1d_L2"] = E.run_conv1d(X, y, g, "L2", seed=a.seed, **ep)
            out["conv1d_as_executed"] = E.run_conv1d(X, y, g, "as-executed", seed=a.seed, **ep)
    if "fourclass" in models:
        rng = np.random.default_rng(a.seed)
        subs4 = cap(range(1, 110))
        tl = [load_fourclass_subject(s, data_path=a.data_path, rng=rng) for s in subs4]
        trials = np.stack([t for t, _ in tl])
        y = np.stack([l for _, l in tl]).reshape(-1, 4).argmax(1)
        g = np.repeat(subs4, trials.shape[1])
        out[f"fourclass_{a.fourclass_protocol}"] = E.run_fourclass(to_64x64(trials), y, g, a.fourclass_protocol,
                                                                   seed=a.seed, **ep)
    print({k: round(v["test_accuracy"], 4) for k, v in out.items()})
    save(out, "benchmark", a.outdir)
