"""Resolve package/data paths. This module does not fit or load models."""
from pathlib import Path
import json, os, sys, tempfile

ROOT = Path(__file__).resolve().parent

def project_paths():
    config = Path(os.environ.get("DUMPWATCH_CONFIG", (ROOT / "config/datasets.local.json" if (ROOT / "config/datasets.local.json").is_file() else ROOT / "config/datasets.template.json")))
    if not config.is_absolute(): config = ROOT / config
    if not config.is_file():
        raise FileNotFoundError(f"Configure {config}; see config/datasets.template.json and SETUP.md")
    values = json.loads(config.read_text(encoding="utf-8"))
    def resolve(key, env):
        value = os.environ.get(env, values.get(key, ""))
        if not value: raise ValueError(f"Set {env} or {key} in {config}")
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    return dict(root=ROOT, task_a_root=resolve("task_a_root", "DUMPWATCH_TASK_A_ROOT"),
                real_zip=resolve("task_b_real_zip", "DUMPWATCH_REAL_ZIP"),
                synthetic_root=resolve("synthetic_root", "DUMPWATCH_SYNTHETIC_ROOT"),
                synthetic_layout=os.environ.get("DUMPWATCH_V1_LAYOUT", values.get("synthetic_layout", "drive")),
                manifest_root=ROOT / "Data/Manifests/Task_B")

def new_output(label):
    parent = os.environ.get("DUMPWATCH_RESULTS")
    if parent: Path(parent).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="dumpwatch_"+label+"_", dir=parent))

def enable_demo_vendor():
    # Bundled native binaries target Windows CPython 3.14 only.
    if sys.platform != "win32" or sys.version_info[:2] != (3, 14): return
    vendor = str(ROOT / "Demo/vendor")
    if vendor not in sys.path: sys.path.insert(0, vendor)

if __name__ == "__main__":
    p = project_paths()
    for key, value in p.items():
        print(key, "=", value, "[exists]" if isinstance(value, Path) and value.exists() else "")
    print("Configuration check only. No training, inference or data copying performed.")
