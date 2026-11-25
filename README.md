# DumpWatch 2.0

**Illegal dumping detection and cleanup verification — CSAI114 Pattern Recognition**

DumpWatch combines video recognition with before/after photograph comparison and an operator-controlled incident registry. This repository contains the existing implementation, selected frozen models, original fitting/evaluation notebooks, experimental evidence and reproducibility tools. Academic reports, presentations, raw datasets, private image outputs and incident records are excluded.

The team collaborated across the complete system. This repository does not assign unverified individual contributions. It is a research prototype: the video and photograph examples depict separate incidents, so the guided handoff is simulated and has **no measured end-to-end accuracy**.

## Recognition tasks and recorded results

### Task A: video detection and onset estimation

The original MIVIA-IWDD-500 snapshot contains **500 videos: 250 dumping and 250 normal**. Its official 400-video training partition was divided into **319 training / 81 development**, with the original **100-video public test partition** held out. The source dataset calls that last directory `VALIDATION_SET`; in this project it is the holdout, not the 81-video development set.

The main pipeline samples **4 FPS**, preserves aspect ratio and pads frames to **224 × 224**, applies ImageNet normalization, then uses frozen **ResNet-18** to produce **512-dimensional frame features**. A trained **64-hidden-unit GRU** feeds two heads: mean-pooled hidden states produce one video-classification logit; frame-wise timing scores estimate onset. The selected checkpoint is **epoch 6**, recorded in `model_lock.json`; seed is **114**.

| Method | Holdout accuracy | Macro-F1 |
|---|---:|---:|
| Training majority baseline | 50.0% | 0.333 |
| Motion features + RBF SVM | 62.0% | 0.597 |
| Frozen ResNet-18 + trained GRU | 75.0% | 0.750 |

The GRU timing-aware event F1 is **0.598**, distinct from classification accuracy. The motion SVM selects C=0.1 on development data. Exact values, per-class precision/recall, labelled confusion matrices, error records, training history and frozen split IDs are in `Reproduction/Task_A/evidence`.

### Task B: cleanup verification

The private collected dataset contains **149 JPEG photographs across 10 locations**, including Waste, Partial and Clean states. Its 37 eligible reviewed pairs are **20 training / 8 development / 9 test**, across **6 / 2 / 2 locations**. Do not confuse photograph counts with eligible pair counts.

Reviewed comparable crops are resized to 128 × 128. Each image has a **240-D descriptor**: 16 spatial cells, each with nine orientation bins and six colour statistics. Reference, inspection, absolute difference and three appearance-score features form **723-D paired inputs**. The selected real-only model uses training-only StandardScaler and LinearSVC (C=0.1, balanced classes, seed 37). Leave-location-out appearance scores protect pair-training features from fitting their own appearance model.

| Retrospective nine-pair comparison | Accuracy | Macro-F1 |
|---|---:|---:|
| Image difference | 55.6% | 0.357 |
| Training majority | 66.7% | 0.400 |
| Compact MLP | 66.7% | 0.400 |
| Forced binary selected SVM | 66.7% | 0.585 |

This comparison is **retrospective**, not a new prospectively selected independent test. The compact MLP uses a different, four-feature representation. The original real-only versus synthetic-V1-assisted development experiment obtained **6/8 versus 6/8**, with no demonstrated improvement. The later local-comparison variant is retained as a separate exploratory experiment.

Operational policy uses the saved, fixed SVM margins:

| Margin / evidence | Recommendation |
|---|---|
| ≤ −0.5 | Waste remains |
| ≥ +0.5 | Apparent clearance |
| Between boundaries, or incomparable footprint | Human review |

On nine test pairs, **4 decisions were accepted, all correct; 5 were deferred; coverage was 44.4%**. This is not 100% overall accuracy. A margin is not a calibrated probability. Even apparent clearance does not automatically close an incident: an independent human confirmation and review note are required.

## Repository map

| Path | Purpose |
|---|---|
| `Demo/app.py`, `workflow.py`, `frozen_preprocessing.py` | Streamlit UI, frozen inference, SQLite workflow and review guards |
| `Demo/models` | Selected trained GRU and Task B SVM; downloaded encoder is excluded |
| `Reproduction/Task_A/task_a.py` | Original preprocessing, classical baseline, GRU training and evaluation |
| `Reproduction/Task_A/evidence` | Frozen model lock, split, metrics, predictions, curves and errors |
| `Training/Task_B/Selected_SVM` | Recovered original selected-SVM fitting/export notebook |
| `Reproduction/Task_B` | Frozen SVM evaluation notebook, model, pair inventories and results |
| `Experiments/Task_B/01_Real_vs_V1` | Original real-only / synthetic-assisted experiment |
| `Experiments/Task_B/02_Local_Comparison` | Separate local-feature comparison |
| `Additional_Evidence/Task_B_Model_Comparison` | Retrospective MLP/difference/majority comparison and frozen coefficients |
| `Evaluation` | No-training evaluation and dataset checks |
| `Data/Manifests/Task_B` | Frozen pair/location manifests and privacy-redacted synthetic allowlist |
| `config`, `project_setup.py` | Relative paths, private configuration override and environment variables |
| `tools` | Encoder acquisition, private demo preparation and split checking |
| `docs` | Source selection, complete file audit, publication changes and test status |

## Windows quick start: inference, without training

The observed successful local runtime is **Windows x64, Python 3.14.0, Streamlit 1.58.0, PyTorch 2.12.1+cpu, torchvision 0.27.1+cpu, NumPy 2.4.6 and Pillow 12.2.0**, with FFmpeg/FFprobe on PATH. The original workstation used a bundled CPython-3.14 torchvision distribution, which is not republished. Installation into a fresh environment from the requirements below is **not claimed as tested**; see `docs/Testing.md` for exact executed checks. Use a compatible official torch/torchvision pair if your platform cannot install the recorded pair.

From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ffmpeg -version
ffprobe -version
.\.venv\Scripts\python.exe -B tools\fetch_encoder.py
.\.venv\Scripts\python.exe -B -m streamlit run Demo\app.py
```

Install FFmpeg separately using its official distribution instructions and ensure **both** executables are on PATH. No administrator rights or training are required by the app. Alternatively activate the virtual environment, then run `Demo/launch.ps1` from PowerShell; it checks dependencies and the local encoder first.

`fetch_encoder.py` obtains the official checkpoint from `https://download.pytorch.org/models/resnet18-f37072fd.pth`, verifies its complete SHA-256, and places it in `Demo/models`. It refuses an incorrect existing file. Offline alternative:

```powershell
python -B tools\fetch_encoder.py --from-file "D:\authorized-local-copy\resnet18-f37072fd.pth"
```

The trained GRU and selected SVM JSON are already included. **Do not rename or replace them.** Their SHA-256 values are checked by the inference code. Only trusted, packaged PyTorch/joblib artifacts should be loaded; pickle-based models can execute code if replaced by untrusted files.

### Using the application

1. In **Detect and register**, upload a short MP4/MOV/MKV/AVI video: nonempty, at most 100 MB and no longer than two minutes. Run live detection. The displayed logit and onset come from the saved GRU, not a cached prediction.
2. A positive result needs operator confirmation before registration. A negative or unchecked result cannot register an incident.
3. In **Verify cleanup**, select the open incident. Upload the waste-present reference and later JPG/PNG photograph. Enter normalized left/top/right/bottom coordinates for comparable visible ground; confirm the footprint.
4. Submit inspection. Read the SVM recommendation and margin. Unconfirmed or uncertain evidence is deferred.
5. In **Review & history**, inspect the evidence. Where appropriate, independently confirm clearance, add a reason and approve final closure. History persists in SQLite.

Raw media and screenshots are not included. The UI defaults to upload controls when examples are absent. After obtaining both datasets through permitted channels, optional `python -B tools/prepare_private_demo.py` creates the **eight original development examples locally**, checks every media hash and refuses conflicting files. These files stay ignored by Git. The clear-photo example itself has an uncertain margin around +0.258: human review is expected.

For rehearsal, isolate the database before launching:

```powershell
$env:DUMPWATCH_DB = Join-Path $env:TEMP ("dumpwatch_rehearsal_" + [guid]::NewGuid().ToString("N") + ".sqlite3")
python -B -m streamlit run Demo\app.py
```

Otherwise the app initializes `Demo/data/incidents.sqlite3` automatically. No private database is distributed. Do not commit newly created media or incident records.

## Dataset configuration: no source edits

Default paths are relative to this repository. `config/datasets.template.json` defines:

```json
{
  "task_a_root": "Data/Task_A",
  "task_b_real_zip": "Data/Real/dumpwatch_real_dataset.zip",
  "synthetic_root": "Data/Synthetic_V1",
  "synthetic_layout": "drive"
}
```

Use these locations directly, or copy the template to **ignored** `config/datasets.local.json` and edit only that private configuration. Environment variables override individual values: `DUMPWATCH_TASK_A_ROOT`, `DUMPWATCH_REAL_ZIP`, `DUMPWATCH_SYNTHETIC_ROOT` and `DUMPWATCH_V1_LAYOUT`. `DUMPWATCH_CONFIG` selects a different JSON configuration. `python -B project_setup.py` prints resolved paths without fitting or loading a model. Paths are resolved from the repository, not your current working directory.

### Task A — official MIVIA-IWDD-500

Official source: **https://github.com/MiviaLab/IllegalWasteDumping**. On 25 November 2026 its README explicitly lists separate Google Drive links for the 400-video training set and 100-video public test set:

- Training: https://drive.google.com/drive/folders/1DgXQnvK3keGP9BKB-cqpixnKaNLl2746
- Public test: https://drive.google.com/drive/folders/1OnA3rHn5uKAMEuHsQc11oxueTT-TQQwJ

The links were followed to pages titled `TRAINING_SET` and `VALIDATION_SET`. A complete independent download and byte-for-byte comparison of those online folders was **not** performed. The videos are not assumed to be stored in GitHub. Use the current official README if links change; contact the dataset authors listed there if access is unavailable.

Download both partitions with their per-video JSON annotations. Prepare these paths (avoid an extra nested `TRAINING_SET/TRAINING_SET`):

| Directory | Required contents |
|---|---|
| `Data/Task_A/TRAINING_SET/videos` | 400 original MP4s |
| `Data/Task_A/TRAINING_SET/labels` | Corresponding original JSON annotations |
| `Data/Task_A/VALIDATION_SET/videos` | 100 original holdout MP4s |
| `Data/Task_A/VALIDATION_SET/labels` | Corresponding original JSON annotations |

Keep original filenames and labels. Run:

```powershell
python -B Data\verify_handoff.py --task a --a-scope all
```

This compares all 500 video hashes and label/onset semantics against the frozen project manifest. Annotation formatting need not be byte-identical. If validation fails, investigate the upstream version; do not alter labels, hashes or splits to make it pass. `Reproduction/Task_A/evidence/split.csv` freezes the 319/81 membership; `decoded_dataset_manifest.csv` includes the 100 holdout IDs. Evaluation uses those locked memberships.

Cite the two publications in `Data/MIVIA_CITATIONS.bib`, as requested by the official repository. Public academic access and citation do not by themselves establish permission to redistribute the videos. This repository does not mirror MIVIA footage or promise anonymous access to every upstream file.

### Task B — private collected dataset

**The original photographs will be shared privately with the instructor through a restricted Google Drive link sent by email. Neither the photographs nor that private link are published here.** Request access directly from the team; there is no public download URL. Private academic marking access does not authorize public redistribution.

Download the exact original **`dumpwatch_real_dataset.zip`** and place it at:

```text
Data/Real/dumpwatch_real_dataset.zip
```

The code reads this archive **without extracting it**. Its original internal structure is `real/location_01/` through `real/location_10/`, containing `B0/` reference photographs and case directories such as `case_01_small/W/`, `case_01_small/A_partial/` and `case_01_small/A_removed/`. Case inventories vary by location; do not create missing folders or substitute photographs. These correspond to waste-present, partial and removed/clean cases. Keep internal filenames and the top-level `dataset_manifest.csv` unchanged. Do not flatten, rename or re-zip it. Expected size is **29,169,726 bytes**, containing 149 JPEGs. Expected archive SHA-256:

```text
1e97cfe51b4ef7fc9e61aaa7cbd2cbce4ac5a24fd274f07503d7deb5138f0412
```

Alternatively set `$env:DUMPWATCH_REAL_ZIP = "D:\private-data\dumpwatch_real_dataset.zip"`. No Python or notebook edits are needed. Validate:

```powershell
python -B Data\verify_handoff.py --task b
```

The repository supplies the reviewed pair inventories, crop coordinates, labels and original location assignments. These are checked against the frozen model, with train/development/test location counts 6/2/2 and eligible pair counts 20/8/9. Do not resplit or move test pairs into training. The new private Drive link must stay in email, not README, configuration committed to Git, notebook outputs or issue comments.

### Optional historical synthetic V1 data

Only the historical synthetic-assisted experiments need this data. The selected real-only SVM and its evaluation do not. The allowlist retains original relative file paths, sizes and hashes; private Drive URLs/IDs are removed. Obtain authorized V1 files privately from the team. Use `Data/Synthetic_V1` with its original `location N/...` structure and `synthetic_layout: drive`, or an archive mirror with `location_01/...` and `synthetic_layout: archive_mirror`. The unchanged model inputs are verified by their image/metadata hashes. Redacting access references changes the allowlist JSON's own byte hash; historical results retain the original hash and are not rewritten.

## Evaluation using saved models, without retraining

Task B needs only the existing real ZIP, NumPy and Pillow:

```powershell
python -B Evaluation\evaluate_frozen.py --task b
```

It freshly evaluates four frozen methods on all eight development and nine previously evaluated test pairs, comparing decisions/scores and metrics with recorded evidence. No `.fit()` is called. This does not turn a retrospective comparison into an independent test.

Task A also needs scikit-learn, joblib, torch/torchvision, the official encoder and FFmpeg. The motion-SVM artifact was saved with **scikit-learn 1.6.1**. Prefer an environment compatible with that original version; `requirements-evaluation.txt` records it. The earlier local full-holdout check used 1.9.1 with a version warning and matched the locked outputs; that does not establish general cross-version pickle compatibility. Do not resave the model merely to remove the warning.

```powershell
python -m pip install -r requirements-evaluation.txt
python -B Evaluation\evaluate_frozen.py --task a
# Optional longer inference: all 100 holdout videos, still no training
python -B Evaluation\evaluate_frozen.py --task a --full-a
```

The default Task A command checks archived metrics for 100 clips **and separately performs fresh inference on four clips**. Those are different checks. The original full 100-video fresh verification is retained under `QA/Validation/Completion`; it matched decisions/onsets and scores within 1e-4 tolerance. Packaging does not claim to have repeated full inference again.

Each evaluation writes a new temporary results directory. `--output` may select a **new, nonexistent** directory; existing output paths are refused. Historical evidence remains unchanged. `Evaluation/verify_datasets.py` additionally checks the synthetic inputs and requires those optional files; use `Data/verify_handoff.py --task b` when only real Task B data is available.

## Original fitting notebooks and optional training

Start Jupyter from the repository root, or set `DUMPWATCH_PACKAGE` to it. Install Jupyter and the appropriate original training dependencies separately. `requirements-training-task-b.txt` and `Reproduction/Task_A/Colab_Runtime_Dependencies.txt` record the original environments; their clean installation has not been demonstrated. Do not assume the CPU inference environment reproduces GPU training.

- **Selected SVM:** open `Training/Task_B/Selected_SVM/dumpwatch_task_b_final_frozen.ipynb`. It loads configured real data, fits appearance/pair SVMs and exports to a new temporary output. Reserved test execution stays disabled by default. It never replaces the submitted model.
- **Task A:** open `Reproduction/Task_A/DumpWatch_Task_A_Colab.ipynb`. Despite its historical name, its portable setup uses `project_setup.py`; GPU/Colab is optional. It invokes the original runner.
- **Synthetic arms:** use the unsuffixed notebooks in the two `Experiments/Task_B` folders. `_executed` companions preserve historical text/numeric outputs with private media removed.
- **Additional comparison:** the unsuffixed notebook under `Additional_Evidence/Task_B_Model_Comparison` contains MLP fitting. Use the frozen evaluation CLI for no-training verification instead.
- **Evaluation-only:** the Task B final-evaluation notebook and `Reproduction/DumpWatch_2.0_Reproduction.ipynb` load saved coefficients; use the CLI as the simpler supported route.

An optional Task A full training command, **not executed in this preparation**, is:

```powershell
python -B Reproduction\Task_A\task_a.py --root "Data\Task_A" --out "results_local\new_task_a" --cache "results_local\features" --mode full
python -B tools\check_training_split.py "results_local\new_task_a\split.csv"
```

The original runner deterministically derives train/development groups using seed 114; compare its resulting split against the retained frozen one before comparing a newly fitted run. Software/data differences may change fitting outcomes. No guarantee of identical retrained coefficients is made, and no models were retrained to assemble this repository.

## Limitations, troubleshooting and verification

- Task A has possible scene overlap between source training and official holdout clips. Do not describe this as a verified unseen-camera evaluation.
- Task B is small, uses staged collected cases and reviewed crops, and may contain related views within locations. Its retrospective comparisons and abstention coverage must remain visible.
- Missing data: run `project_setup.py`, check configuration and validate the private archive. Raw datasets are intentionally excluded.
- Missing checkpoint: run `tools/fetch_encoder.py`; selected task-specific weights are already present and checksum-pinned.
- Torchvision import error: use a supported official torch/torchvision/Python combination. The original Windows vendor distribution is not a hidden runtime requirement in the public launch script.
- FFmpeg error: confirm both executable names work in the same terminal that launches Streamlit.
- No examples in the UI: expected for a public checkout; upload authorized media, or privately prepare the original examples locally.
- Notebook fit cells are not demo startup steps. Training may be expensive; do not run them just to launch the app.

See **[docs/Testing.md](docs/Testing.md)** for PASS/FAIL/BLOCKED/NOT RUN results and observed dependencies; **[docs/Source_Selection.md](docs/Source_Selection.md)** for notebook selection; and **[docs/File_Audit.csv](docs/File_Audit.csv)** for original-file decisions. Source availability, successful frozen evaluation and full training reproducibility are reported separately.

Third-party attribution is in `Attribution/CODE_SOURCES.md` and `THIRD_PARTY_NOTICES.md`. Dataset rights are separate from software availability. No additional software licence has been assigned on behalf of the team. AI assistance was used for recovery, packaging, setup documentation and verification; no individual authorship or development contribution is fabricated.
