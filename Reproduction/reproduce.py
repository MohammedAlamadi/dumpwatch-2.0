"""Frozen evaluation and separate archived prediction checks."""
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def compare(actual, expected, keys):
    for key in keys:
        a, e = actual[key], expected[key]
        if not (math.isclose(a, e, abs_tol=1e-9) if isinstance(e, float) else a == e):
            raise AssertionError(f"{key}: computed {a!r}; archived {e!r}")

A_KEYS = ("n", "accuracy", "macro_f1", "matrix", "event_f1", "event_tp", "event_fp",
          "event_fn", "event_window_s", "timing_eligible", "timing_reported", "onset_mae_s",
          "onset_median_ae_s", "signed_mean_error_s", "excluded_timing_ids")
B_KEYS = ("n", "confusion_matrix", "accepted", "correct_accepted", "abstained",
          "coverage", "accepted_accuracy", "false_clearance_count", "false_alarm_count")

def task_a(root, data_root, results):
    import task_a as a
    import torch
    run = root / "evidence"
    frozen = json.loads((run / "model_lock.json").read_text())
    for name in ("gru.pt", "split.csv"):
        a.require(a.sha(run / name) == frozen["locked_files"][name], f"Frozen file changed: {name}")
    a.require(frozen["classification_threshold"] == 0 and frozen["event_window"] == [-5, 10], "Policy changed")
    split = {r["id"]: r for r in read_csv(run / "decoded_dataset_manifest.csv") if r["assignment"] == "holdout"}
    records = [json.loads(s) for s in (run / "holdout_ingest.jsonl").read_text().splitlines()]
    a.require(len(split) == len(records) == 100 and set(split) == {r["id"] for r in records},
              "Task A holdout membership mismatch")
    for r in records:
        a.require(split[r["id"]]["sha256"] == r["sha"] and split[r["id"]]["event"] == r["event"],
                  f"Holdout metadata changed: {r['id']}")
    reference = json.loads((run / "holdout_metrics.json").read_text())["resnet18_gru"]
    archived = {r["id"]: r for r in read_csv(run / "holdout_predictions.csv") if r["method"] == "resnet18_gru"}
    prior = [(int(archived[r["id"]]["prediction"]),
              float(archived[r["id"]]["predicted_onset_s"]) if archived[r["id"]]["predicted_onset_s"] else None,
              float(archived[r["id"]]["score"])) for r in records]
    compare(a.measure(records, prior), reference, A_KEYS)
    print("Task A: recalculated archived metrics from saved predictions.")

    candidates = [data_root / "VALIDATION_SET" / "videos", data_root / "videos", data_root]
    folder = next((p for p in candidates if (p / f"{records[0]['id']}.mp4").is_file()), candidates[0])
    missing = [r["id"] for r in records if not (folder / f"{r['id']}.mp4").is_file()]
    if missing:
        print(f"Task A fresh inference blocked: {len(missing)} locked holdout videos missing from {folder}.")
        return None
    for r in records:
        r["path"] = folder / f"{r['id']}.mp4"
        a.require(a.sha(r["path"]) == r["sha"], f"Task A video bytes differ: {r['id']}")
    from torchvision.models import resnet18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.manual_seed(a.SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(a.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    weights = root.parents[1] / "Demo/models/resnet18-f37072fd.pth"
    a.require(a.sha(weights) == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
              "Encoder weights differ")
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    net = net.eval().requires_grad_(False).to(device)
    gru = a.GRU().to(device)
    gru.load_state_dict(torch.load(run / "gru.pt", map_location=device, weights_only=True))
    timing_flags = [r["timing_valid"] for r in records]
    with TemporaryDirectory(prefix="dumpwatch_a_") as directory:
        a.ingest(records, Path(directory), net, device, results / "task_a_ingest.jsonl")
        a.require([r["timing_valid"] for r in records] == timing_flags, "Timing eligibility changed")
        predictions = a.gru_predictions(gru, records, device)
    actual = a.measure(records, predictions)
    (results / "task_a_fresh_metrics.json").write_text(json.dumps(actual, indent=2), encoding="utf-8")
    a.write_csv(results / "task_a_fresh_predictions.csv", [
        dict(id=r["id"], actual=r["y"], prediction=y, onset_s=t, raw_score=s)
        for r, (y, t, s) in zip(records, predictions, strict=True)
    ])
    compare(actual, reference, A_KEYS)
    return actual

def task_b(root, real_zip, results):
    original = json.loads((root / "dumpwatch_task_b_final_evaluation.ipynb").read_text(encoding="utf-8"))
    def source(i):
        return "".join(original["cells"][i]["source"])
    pins = "\n".join(line for line in source(1).splitlines() if line.startswith(
        ("EXPECTED_MODEL_SHA256", "EXPECTED_TEST_CSV_SHA256", "EXPECTED_DEV_CSV_SHA256", "PREPROCESSING_SHA256")
    ))
    scoring = source(5).split("\nchart=Image.new", 1)[0]
    definitions = scoring.split("\noutput=OUTPUT_ROOT", 1)[0]
    namespace = dict(REAL_ZIP=real_zip, OUTPUT_ROOT=results / "task_b",
                     SPLIT_CSV=root / "real_location_split.csv",
                     FROZEN_DEV_CSV=root / "frozen_train_development_pair_inventory.csv",
                     MODEL_JSON=root / "frozen_real_only_svm.json",
                     LOCK_JSON=root / "frozen_test_inventory_lock.json",
                     TEST_CSV=root / "frozen_test_pair_inventory.csv")
    for code in (pins, source(3), definitions):
        exec(code, namespace)
    reference = json.loads((root / "results/final_test_metrics.json").read_text())
    saved = read_csv(root / "results/final_test_predictions.csv")
    for key in ("forced", "fixed"):
        compare(namespace["metrics"](saved, f"{key}_prediction"), reference[key], B_KEYS)
    print("Task B: recalculated archived metrics from saved predictions.")
    if not real_zip.is_file():
        print(f"Task B fresh inference blocked: authorized archive missing at {real_zip}.")
        return None
    exec(source(4), namespace)
    exec(scoring, namespace)
    result = namespace["result"]
    for key in ("forced", "fixed"):
        compare(result[key], reference[key], B_KEYS)
    for a, b in zip(namespace["predictions"], saved, strict=True):
        for key in ("pair_id", "forced_prediction", "fixed_prediction"):
            assert a[key] == b[key], (key, a["pair_id"])
    return result

