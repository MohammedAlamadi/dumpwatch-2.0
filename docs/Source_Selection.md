# Authoritative source selection

Task A: `Reproduction/Task_A/task_a.py` is the executable original runner. Its model lock selects GRU epoch 6, hidden size 64, seed 114 and motion SVM C=0.1. The portable training notebook invokes that runner; the old encoded Colab bootstrap is excluded. Frozen partitions, checkpoint, history, predictions and confusion matrices remain.

Task B: `Training/Task_B/Selected_SVM/dumpwatch_task_b_final_frozen.ipynb` contains selected real-only SVM fitting/export. Its scientific function cells are checked against the recovered original. `Reproduction/Task_B/frozen_real_only_svm.json` is the selected artifact; copies required by inference/comparison are retained byte-for-byte.

Synthetic experiments: the portable notebooks in `Experiments/Task_B/01_Real_vs_V1` and `02_Local_Comparison` are executable fitting sources. Their `_executed` companions are historical evidence, not alternate selected models. Rich image outputs and private-user path strings are removed from publication copies; originals remain in the preserved source archive. Numeric/text results remain. The original development comparison was 6/8 versus 6/8.

Additional MLP/image-difference/majority comparison: `Additional_Evidence/Task_B_Model_Comparison/dumpwatch_task_b_comparison.ipynb` and `recorded_run` retain implementation and coefficients. This nine-pair comparison is retrospective. No fitting was performed during packaging.

Public-only changes: optional sample availability in the Streamlit UI; relative configuration fallback; dependency/launch documentation. Workflow, preprocessing, frozen weights, labels and locked split files are not changed. Original scientific source is identified in the file audit by its SHA-256.

Excluded content remains in the untouched original archive. `File_Audit.csv` gives the decision for every original file. `Publication_Changes.json` identifies transformations. No academic report/presentation, raw image/video, private link, incident database or vendored native dependency is published.
