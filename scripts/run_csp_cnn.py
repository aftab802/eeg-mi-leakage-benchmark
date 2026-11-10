"""Filter-bank CSP + compact CNN (left vs right fist imagery).

Data: all 109 subjects, R04/R08/R12, resampled to 160 Hz, 8-30 Hz, 0-4 s
epochs (641 samples). CSP filters and the feature scaler are fitted on the
training partition only. --protocol L0 (default) splits by subject; --protocol
L2 reproduces a class-stratified, record-level split.
"""
from _common import parser, save, subjects_arg

from cybex_eeg.mi_experiments import run_csp_cnn
from cybex_eeg.physionet import load_lr_dataset

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--subjects", default="1-109")
    p.add_argument("--protocol", default="L0", choices=["L0", "L2"])
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args()
    X, y, g = load_lr_dataset(subjects_arg(a.subjects), tmin=0.0, tmax=4.0, resample=160.0, data_path=a.data_path)
    print("Data:", X.shape, "subjects:", len(set(g)))
    save(run_csp_cnn(X, y, g, protocol=a.protocol, epochs=a.epochs, seed=a.seed), f"csp_cnn_{a.protocol}", a.outdir)
