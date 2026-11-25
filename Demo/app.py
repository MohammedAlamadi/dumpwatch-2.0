"""Run with streamlit run app.py."""
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw
import streamlit as st

from workflow import ROOT, Registry, detect_video, inspect_pair

st.set_page_config(page_title="DumpWatch 2.0", page_icon="♻️", layout="wide")
examples = json.loads((ROOT / "examples" / "manifest.json").read_text(encoding="utf8"))
registry = Registry(os.getenv("DUMPWATCH_DB", ROOT / "data" / "incidents.sqlite3"))
errors = (OSError, ValueError, RuntimeError, subprocess.SubprocessError)

st.title("DumpWatch 2.0")
st.caption("Research prototype · Detection to cleanup review")
st.info("Guided examples simulate an operational handoff. The MIVIA clip and the separate cleanup photographs depict different incidents. The demo supplies no end-to-end accuracy estimate.")
with st.sidebar:
    st.header("Guided walkthrough")
    st.markdown("1. Run **Dumping example** and confirm registration.\n"
                "2. Submit **Waste remains example** to update the incident.\n"
                "3. Submit **Needs review example** to show abstention.\n"
                "4. Submit **Closure review example**. Inspect the photographs, then explicitly approve closure in **Review & history**.")
    st.caption("Try the no-dumping example to see why a negative prediction cannot create an incident. Uploads stay in the current session. The local registry keeps metadata and decisions only.")
detect, clean, history = st.tabs(["1 · Detect and register", "2 · Verify cleanup", "3 · Review & history"])

with detect:
    st.header("Video detection")
    video_choices = [label for label,key in [("Dumping example","detected"),("No-dumping example","negative")] if (ROOT / "examples" / examples[key]["file"]).is_file()] + ["Upload video"]
    source = st.radio("Video source", video_choices, horizontal=True)
    sample = source != "Upload video"
    if sample:
        item = examples["detected" if source == "Dumping example" else "negative"]
        filename = item["file"]
        raw = (ROOT / "examples" / filename).read_bytes()
    else:
        uploaded = st.file_uploader("Short video, up to two minutes", type=["mp4", "mov", "mkv", "avi"])
        raw, filename = (uploaded.getvalue(), uploaded.name) if uploaded else (None, "")
    simulated = True if sample else st.checkbox("Register as a simulated incident", value=True)
    key = (sha256(raw).hexdigest(), filename, simulated) if raw else None
    left, right = st.columns([3, 2], gap="large")
    with left:
        if raw:
            st.video(raw)
        submit = st.button("Run live detection", type="primary", disabled=not raw)
    if submit:
        st.session_state.pop("detection", None)
        try:
            with st.spinner("Decoding and running the frozen video model…"):
                detection, frame = detect_video(raw, filename, simulated)
            st.session_state.detection = (key, detection, frame)
            st.session_state.pop("registered_detection", None)
        except errors as exc:
            st.error(f"Detection could not run. Check the file and FFmpeg installation. {exc}")
    saved = st.session_state.get("detection")
    if saved and saved[0] == key:
        _, result, frame = saved
        with right:
            st.subheader("Live model result")
            if result["detected"]:
                st.success("Dumping detected · Awaiting confirmation")
            else:
                st.info("Dumping not detected")
            st.metric("Detection score (logit)", f'{result["score"]:+.3f}')
            if result["detected"]:
                st.caption(f'Estimated onset {result["onset_seconds"]:.2f} s after clip start')
            st.image(frame, caption="Estimated onset frame" if result["detected"] else "First sampled frame")
            pending = st.session_state.get("registered_detection")
            if pending and pending[0] == key:
                st.success(f"Registered {pending[1]}")
            elif result["detected"]:
                with st.form("register"):
                    camera = st.text_input("Camera or location ID (optional)")
                    confirmed = st.checkbox("I confirm this detection for registration")
                    register = st.form_submit_button("Register incident")
                if register:
                    try:
                        incident = registry.register(result, camera, confirmed)
                        st.session_state.registered_detection = (key, incident)
                        st.rerun()
                    except errors as exc:
                        st.error(str(exc))


def annotated(raw, roi):
    with Image.open(BytesIO(raw)) as source:
        source.thumbnail((700, 450))
        canvas = source.convert("RGB")
    w, h = canvas.size
    l, t, r, b = roi
    ImageDraw.Draw(canvas).rectangle((l*w,t*h,r*w,b*h), outline="#24e6aa", width=3)
    return canvas


def roi_editor(name, caption):
    st.caption(caption)
    left, top, right, bottom = st.columns(4)
    values = []
    for position, column, default in zip(("Left", "Top", "Right", "Bottom"), (left,top,right,bottom), (0.,0.,1.,1.)):
        with column:
            values.append(st.number_input(position, min_value=0., max_value=1., value=default,
                                          step=.01, key=f"{name}_{position}"))
    return values


with clean:
    st.header("Cleanup inspection")
    open_cases = [r for r in registry.list() if r["status"] != "Verified clean"]
    if not open_cases:
        st.info("Register a detected incident to submit an inspection.")
    else:
        incident_id = st.selectbox("Open incident", [r["id"] for r in open_cases])
        record = next(r for r in open_cases if r["id"] == incident_id)
        sample_keys = {"Waste remains example": "remains", "Needs review example": "review",
                       "Closure review example": "review_closure"}
        available_pairs = [label for label,key in sample_keys.items() if all((ROOT / "examples" / examples[key][name]).is_file() for name in ("W","inspection"))]
        choices = [*available_pairs, "Upload photographs"] if record["simulated"] else ["Upload photographs"]
        option = st.radio("Inspection source", choices, horizontal=True)
        sample_pair = option != "Upload photographs"
        a, b = st.columns(2)
        w, later, names = None, None, ("", "")
        w_roi, later_roi = [0.,0.,1.,1.], [0.,0.,1.,1.]
        if sample_pair:
            item = examples[sample_keys[option]]
            names = (item["W"], item["inspection"])
            w, later = [(ROOT / "examples" / n).read_bytes() for n in names]
            w_roi, later_roi = item["W_roi"], item["inspection_roi"]
        else:
            with a:
                first = st.file_uploader("Waste-present reference (W)", type=["jpg","jpeg","png"], key="upload_w")
                w_roi = roi_editor("w", "Reference crop · normalized coordinates")
            with b:
                second = st.file_uploader("Later inspection", type=["jpg","jpeg","png"], key="upload_inspection")
                later_roi = roi_editor("inspection", "Inspection crop · normalized coordinates")
            if first and second:
                w, later, names = first.getvalue(), second.getvalue(), (first.name, second.name)
        ready = False
        if w and later:
            try:
                with a:
                    st.image(annotated(w,w_roi), caption="Waste-present reference (W)")
                with b:
                    st.image(annotated(later,later_roi), caption="Later inspection")
                ready = True
            except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
                st.error("Check both image files and crop coordinates.")
        assessable = st.checkbox("I confirm that the crops show the same visible ground footprint",
                                 key=f"coverage_{incident_id}_{option}")
        digests = (sha256(w).hexdigest(), sha256(later).hexdigest()) if ready else None
        inspection_key = (incident_id, names, digests, tuple(w_roi), tuple(later_roi), assessable)
        if st.button("Submit inspection", type="primary", disabled=not ready):
            try:
                with st.spinner("Running the frozen clearance policy…"):
                    decision = inspect_pair(w,later,w_roi,later_roi,assessable)
                status = registry.inspection(incident_id,decision,{
                    "sample":sample_pair,"simulated_handoff":bool(record["simulated"]),
                    "source":names,"reference_sha256":digests[0],
                    "inspection_sha256":digests[1],
                    "reference_roi":w_roi,"inspection_roi":later_roi,"visible_footprint_confirmed":assessable,
                })
                st.session_state.latest = (inspection_key, status, decision)
            except errors as exc:
                st.error(f"Inspection could not run. {exc}")
        latest = st.session_state.get("latest")
        if latest and latest[0] == inspection_key:
            _, status, decision = latest
            if status == "Pending human review":
                st.warning("Requires human review")
            else:
                st.info(status)
            if decision["score"] is not None:
                st.caption(f'SVM clearance margin {decision["score"]:+.3f} · fixed boundaries −0.5 and +0.5')


with history:
    st.header("Incident registry")
    cases = registry.list()
    if not cases:
        st.info("No incidents registered.")
    else:
        st.table([{"Incident":r["id"],"Status":r["status"],"Camera":r["camera"] or "—",
                   "Workflow":"Simulated" if r["simulated"] else "Operator supplied",
                   "Updated (UTC)":r["updated"]} for r in cases])
        record_id = st.selectbox("Review incident", [r["id"] for r in cases])
        record = next(r for r in cases if r["id"] == record_id)
        events = registry.history(record_id)
        info = json.loads(record["detection"])
        st.write(f'{record["id"]} · {record["status"]}')
        st.caption(f'{info["video"]} · registered detection {info["detection_time_utc"]} · score {info["score"]:+.3f} · onset {info["onset_seconds"]:.2f} s')
        st.table([{"Status":e["status"],"Time (UTC)":e["time"]} for e in events])
        with st.expander("Recorded evidence"):
            st.json({"detection":info,"events":[dict(e,detail=json.loads(e["detail"])) for e in events]})
        if record["status"] in ("Pending human review", "Cleanup appears complete"):
            with st.form(f"closure_{record_id}_{events[-1]['seq']}"):
                verified = st.checkbox("I independently verified that the original waste footprint is clear")
                note = st.text_area("Review note")
                approve = st.form_submit_button("Approve final closure")
            if approve:
                try:
                    registry.close(record_id,events[-1]["seq"],verified,note)
                    st.session_state.pop("latest",None)
                    st.rerun()
                except errors as exc:
                    st.error(str(exc))
