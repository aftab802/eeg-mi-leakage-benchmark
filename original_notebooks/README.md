# Original notebooks

These are the Google Colab notebooks and scripts the reported results were first produced with, kept with their printed outputs as a provenance record. Paths point to the authors' Google Drive and must be edited to run them. The maintained, tested implementation is in `cybex_eeg/` and `scripts/`.

* `group_aware_eegnet_original.ipynb` — originally saved as “EEG Motor Imagery CNN.ipynb”.
* `four_class_cnn_original.ipynb` — originally saved as “EEG-Based Differential Diagnosis of Alzheimer's Disease and Frontotemporal Dementia Using Graph Signal Processing Features and XGBoost.ipynb”.
* `early_colab_pipeline/` — early Colab pipeline (EEG-DL-derived data loader, metrics, GCN driver). Its `eegnet_full_pipeline.py` splits pooled trials with `train_test_split`, i.e. the record-level protocol later replaced by the subject-wise split.
