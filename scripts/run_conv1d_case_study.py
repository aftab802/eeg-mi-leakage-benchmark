"""1D CNN protocol-leakage case study.

Data: all 109 subjects, runs R03/R07/R11 (executed) + R04/R08/R12 (imagined),
left vs right fist, 0-2 s epochs (320 samples at 160 Hz), 8-30 Hz.

--protocol as-executed  record-level 80/20 split stratified by class, test set
                        reused as validation data (the 71.46 % protocol)
--protocol L2           record-level split with a separate validation split
--protocol L0           subject-wise split with intersection check (corrected)
"""
from _common import parser, save, subjects_arg

from cybex_eeg.mi_experiments import run_conv1d
from cybex_eeg.physionet import EXECUTED_LR_RUNS, IMAGERY_LR_RUNS, load_lr_dataset

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--subjects", default="1-109")
    p.add_argument("--protocol", default="as-executed", choices=["as-executed", "L2", "L0"])
    p.add_argument("--epochs", type=int, default=40)
    a = p.parse_args()
    X, y, g = load_lr_dataset(subjects_arg(a.subjects), runs=EXECUTED_LR_RUNS + IMAGERY_LR_RUNS, tmin=0.0,
                              tmax=2.0 - 1 / 160, resample=160.0, data_path=a.data_path)
    print("Data:", X.shape, "subjects:", len(set(g)))
    save(run_conv1d(X, y, g, protocol=a.protocol, epochs=a.epochs, seed=a.seed), f"conv1d_{a.protocol}", a.outdir)
