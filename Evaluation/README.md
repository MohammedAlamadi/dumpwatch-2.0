# Frozen evaluation

From the repository root, configure the original datasets and run `python -B Evaluation/evaluate_frozen.py --task b` or `--task a`. Task B freshly checks eight development and nine retrospective test pairs across four frozen methods. Task A defaults to four fresh clips plus a separate archived-100-metrics check; `--full-a` performs longer fresh inference on all 100 holdout clips. No models are fitted. Existing output directories are refused.

The repository README describes dependencies, original sklearn 1.6.1 pickle compatibility, exact data access, optional synthetic checks and known limitations. `docs/Testing.md` distinguishes current short checks from retained historical full evaluation and unperformed fresh-environment training.
