"""Local inference and incident storage for the research demonstration."""
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import json
import sqlite3

import numpy as np
import torch

from frozen_preprocessing import FPS, GRU, decode, embeddings, frozen_pair_features, frozen_score, descriptor, probe, select_clearance

ROOT = Path(__file__).resolve().parent
MODELS = {
    "gru.pt": "0e96cfb5077dc00707a1e7bbf53d7dde6d99671d661d46d46f96847c9901c263",
    "frozen_real_only_svm.json": "b6bc6391b3af0e7ea2999e0702876c1afa980f51f83470fcce4238b430f0b6b5",
    "resnet18-f37072fd.pth": "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
}
OUTCOMES = {
    "waste_still_present": "Waste remains",
    "fully_cleared": "Cleanup appears complete",
    "abstain": "Pending human review",
}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def model_path(name):
    path = ROOT / "models" / name
    if sha256(path.read_bytes()).hexdigest() != MODELS[name]:
        raise ValueError(f"Frozen model checksum differs: {name}")
    return path


@lru_cache(maxsize=1)
def video_models():
    from torchvision.models import resnet18
    torch.set_num_threads(2)
    net = resnet18(weights=None)
    net.load_state_dict(torch.load(model_path("resnet18-f37072fd.pth"), map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    gru = GRU()
    gru.load_state_dict(torch.load(model_path("gru.pt"), map_location="cpu", weights_only=True))
    return net.eval().requires_grad_(False), gru.eval().requires_grad_(False)


@lru_cache(maxsize=1)
def cleanup_model():
    return json.loads(model_path("frozen_real_only_svm.json").read_text(encoding="utf8"))


def detect_video(raw, filename, simulated=False):
    if not raw or len(raw) > 100 * 1024**2:
        raise ValueError("Choose a nonempty video of at most 100 MB.")
    net, gru = video_models()
    with TemporaryDirectory(prefix="dumpwatch-video-") as directory:
        video = Path(directory) / "clip.mp4"
        video.write_bytes(raw)
        duration, _, _ = probe(video)
        if not np.isfinite(duration) or not 0 < duration <= 120:
            raise ValueError("Choose a clip between 0 and 120 seconds.")
        frames = decode(video)
    times = np.arange(len(frames), dtype=np.float32) / FPS
    valid = times < duration
    times, frames = times[valid], frames[valid]
    if not len(frames):
        raise ValueError("The clip contains no usable sampled frames.")
    with torch.inference_mode():
        x = torch.from_numpy(embeddings(frames, net, torch.device("cpu"))).unsqueeze(0)
        score, timing = gru(x, torch.tensor([len(frames)]))
    logit = float(score.item())
    if not np.isfinite(logit):
        raise ValueError("The detector produced an invalid score.")
    detected = logit >= 0
    index = int(timing.argmax()) if detected else 0
    return {
        "detected": detected, "score": logit,
        "onset_seconds": float(times[index]) if detected else None,
        "video": Path(filename).name, "source_sha256": sha256(raw).hexdigest(),
        "detection_time_utc": utc(), "simulated": bool(simulated),
    }, frames[index]


def inspect_pair(waste, inspection, w_roi, a_roi, assessable):
    if not assessable:
        return {"outcome": "abstain", "score": None,
                "reason": "Reviewer did not confirm a comparable visible footprint."}
    model = cleanup_model()
    w = descriptor(waste, w_roi)
    a = descriptor(inspection, a_roi)
    score = float(frozen_score(frozen_pair_features(w, a, model), model["pair"]))
    if not np.isfinite(score):
        raise ValueError("The clearance model produced an invalid score.")
    policy = model["policy"]
    outcome = str(select_clearance(score, True, policy["low"], policy["high"]).item())
    return {"outcome": outcome, "score": score, "reason": "Frozen SVM margin and abstention policy."}


class Registry:
    """Persist incident records and transitions without retaining uploaded media."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS incidents (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, camera TEXT NOT NULL,
                    simulated INTEGER NOT NULL, detection TEXT NOT NULL, updated TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY, incident TEXT NOT NULL REFERENCES incidents(id),
                    status TEXT NOT NULL, time TEXT NOT NULL, detail TEXT NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    def register(self, detection, camera, confirmed):
        if not confirmed or not detection["detected"]:
            raise ValueError("Confirm a positive detection before registering it.")
        incident = "DW-" + uuid4().hex[:12].upper()
        stamp = utc()
        with self.connection() as db:
            db.execute("INSERT INTO incidents VALUES (?, ?, ?, ?, ?, ?)",
                       (incident, "Awaiting cleanup", camera.strip(), detection["simulated"],
                        json.dumps(detection), stamp))
            self.event(db, incident, "Registered", {"confirmed_by_operator": True})
            self.event(db, incident, "Awaiting cleanup", {})
        return incident

    @staticmethod
    def event(db, incident, status, detail):
        db.execute("INSERT INTO events(incident,status,time,detail) VALUES (?,?,?,?)",
                   (incident, status, utc(), json.dumps(detail)))

    def list(self):
        with self.connection() as db:
            return [dict(r) for r in db.execute("SELECT * FROM incidents ORDER BY updated DESC, id")]

    def history(self, incident):
        with self.connection() as db:
            return [dict(r) for r in db.execute(
                "SELECT * FROM events WHERE incident=? ORDER BY seq", (incident,))]

    def inspection(self, incident, result, evidence):
        status = OUTCOMES[result["outcome"]]
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            record = db.execute("SELECT status,simulated FROM incidents WHERE id=?", (incident,)).fetchone()
            if not record or record["status"] == "Verified clean":
                raise ValueError("Choose an open incident.")
            if evidence.get("sample") and not record["simulated"]:
                raise ValueError("Example photographs require a simulated incident.")
            self.event(db, incident, "Cleanup inspection received", evidence)
            self.event(db, incident, status, result)
            db.execute("UPDATE incidents SET status=?,updated=? WHERE id=?", (status, utc(), incident))
        return status

    def close(self, incident, expected_event, confirmed, note):
        if not confirmed or not note.strip():
            raise ValueError("Independent human confirmation and a review note are required.")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            latest = db.execute("SELECT seq,status FROM events WHERE incident=? ORDER BY seq DESC LIMIT 1",
                                (incident,)).fetchone()
            if not latest or latest["seq"] != expected_event or latest["status"] not in (
                "Pending human review", "Cleanup appears complete"
            ):
                raise ValueError("This inspection is no longer eligible for closure.")
            self.event(db, incident, "Verified clean",
                       {"human_approval": True, "review_note": note.strip()})
            db.execute("UPDATE incidents SET status=?,updated=? WHERE id=?", ("Verified clean", utc(), incident))
