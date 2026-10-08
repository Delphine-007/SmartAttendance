"""
app.py - Face-recognition attendance system for managers (Streamlit)
Run:  streamlit run app.py
"""
import datetime as dt
import hashlib
import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import face_engine as fe

DATA = Path("data"); DATA.mkdir(exist_ok=True)
GALLERY_FILE = DATA / "gallery.pkl"
LOG_FILE = DATA / "attendance.csv"
PHOTO_DIR = DATA / "staff_photos"
PHOTO_DIR.mkdir(exist_ok=True)
ATTENDANCE_COLUMNS = [
    "date", "time", "person_id", "name", "status", "similarity",
]
CFG_FILE = Path("models/config.json")
cfg = json.loads(CFG_FILE.read_text()) if CFG_FILE.exists() else {"threshold": 0.60}

st.set_page_config(page_title="Smart Attendance", page_icon="🧑‍💼", layout="wide")

# ----------------------------- storage helpers -------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("numpy._core."):
                module = module.replace("numpy._core.", "numpy.core.", 1)
                return super().find_class(module, name)
            raise


def load_gallery():
    if GALLERY_FILE.exists():
        return _NumpyCompatUnpickler(io.BytesIO(GALLERY_FILE.read_bytes())).load()
    return {}          # {person_id: {"name": str, "emb": ndarray (k,512)}}


def save_gallery(g):
    GALLERY_FILE.write_bytes(pickle.dumps(g))


def staff_photo_path(pid):
    photo_id = hashlib.sha256(pid.encode("utf-8")).hexdigest()
    return PHOTO_DIR / f"{photo_id}.jpg"


def gallery_matrix(g):
    rows, owners = [], []
    for pid, d in g.items():
        rows.append(d["emb"]); owners += [pid] * len(d["emb"])
    return (np.vstack(rows) if rows else None), owners


def load_log():
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE, dtype=str)
        if "status" not in df.columns:
            df["status"] = "Present"
    else:
        df = pd.DataFrame(columns=ATTENDANCE_COLUMNS)
    for column in ATTENDANCE_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[ATTENDANCE_COLUMNS].fillna("")


def save_attendance_statuses(records, day):
    if day.weekday() >= 5:
        raise ValueError("Attendance can only be recorded Monday through Friday.")

    df = load_log()
    now = dt.datetime.now().strftime("%H:%M:%S")
    for record in records:
        status = record["status"]
        if status not in {"Present", "Absent"}:
            raise ValueError(f"Unsupported attendance status: {status}")

        person_id = record["person_id"]
        mask = (df["date"] == str(day)) & (df["person_id"] == person_id)
        previous = df.loc[mask].iloc[-1] if mask.any() else None
        if status == "Present":
            time = now if previous is None or previous["status"] != "Present" else previous["time"]
            similarity = record.get("similarity")
            if similarity is None and previous is not None and previous["status"] == "Present":
                similarity = previous["similarity"]
            similarity = f"{float(similarity):.3f}" if similarity not in (None, "") else ""
        else:
            time, similarity = "", ""

        row = {
            "date": str(day),
            "time": time,
            "person_id": person_id,
            "name": record["name"],
            "status": status,
            "similarity": similarity,
        }
        if previous is None:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df.loc[mask, ATTENDANCE_COLUMNS] = list(row.values())

    df[ATTENDANCE_COLUMNS].to_csv(LOG_FILE, index=False)


gallery = load_gallery()

# ----------------------------- sidebar ---------------------------------------
st.sidebar.title("🧑‍💼 Smart Attendance")
page = st.sidebar.radio("Menu", ["Take attendance", "Enrol staff", "Reports"])
st.sidebar.caption(f"Model: FaceNet (pretrained) · enrolled: {len(gallery)}")
st.sidebar.info("Face embeddings and enrolment photos are stored locally. "
                "Enrol only people who gave consent.")

# ----------------------------- ENROL -----------------------------------------
if page == "Enrol staff":
    st.header("Enrol a staff member")
    c1, c2 = st.columns(2)
    name = c1.text_input("Full name")
    pid = c2.text_input("Staff ID")
    consent = st.checkbox("This person has consented to face-based attendance.")
    mode = st.radio("Photos from", ["Upload (3-5 photos)", "Camera (one shot)"],
                    horizontal=True)
    if mode.startswith("Upload"):
        files = st.file_uploader("Clear face photos, different angles/lighting",
                                 type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    else:
        shot = st.camera_input("Take photo")
        files = [shot] if shot else []

    if st.button("Save enrolment", type="primary"):
        if not (name and pid and consent and files):
            st.error("Name, ID, consent and at least one photo are required.")
        else:
            embs = []
            profile_image = None
            for f in files:
                try:
                    img = Image.open(f).convert("RGB")
                except OSError as exc:
                    st.error(f"Could not read {getattr(f, 'name', 'uploaded photo')}: {exc}")
                    continue
                boxes, faces = fe.detect_faces(img)
                if len(boxes) == 0:
                    st.warning(f"No face found in {getattr(f, 'name', 'photo')}")
                    continue
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                embs.append(fe.embed_tensors(faces[[int(area.argmax())]])[0])
                if profile_image is None:
                    profile_image = img
            if embs:
                old = gallery.get(pid, {}).get("emb", np.empty((0, 512)))
                gallery[pid] = {
                    "name": name,
                    "emb": np.vstack([old, np.array(embs)]),
                }
                profile_image.save(staff_photo_path(pid), format="JPEG", quality=90)
                save_gallery(gallery)
                st.success(f"Enrolled {name} with {len(embs)} new face sample(s).")
            else:
                st.error("No usable face detected. Try clearer photos.")

    if gallery:
        st.subheader("Enrolled staff")
        staff_roster = pd.DataFrame([
            {"person_id": person_id, "name": details["name"]}
            for person_id, details in gallery.items()
        ])
        st.dataframe(staff_roster, hide_index=True)
        rm = st.selectbox("Remove a person (right to be forgotten)",
                          [""] + list(gallery.keys()))
        if rm and st.button("Delete"):
            gallery.pop(rm)
            photo_path = staff_photo_path(rm)
            if photo_path.exists():
                photo_path.unlink()
            save_gallery(gallery)
            st.rerun()

# ----------------------------- ATTENDANCE ------------------------------------
elif page == "Take attendance":
    st.header("Take attendance")
    today = dt.date.today()
    if today.weekday() >= 5:
        st.warning("Attendance can only be recorded Monday through Friday.")
        st.stop()
    st.caption(f"Attendance date: {today:%A, %B %d, %Y}")

    G, owners = gallery_matrix(gallery)
    if G is None:
        st.warning("Enrol staff first.")
        st.stop()
    thr = st.sidebar.slider("Match threshold", 0.30, 0.95,
                            float(cfg["threshold"]), 0.01,
                            help="Higher = stricter (fewer false matches, more 'Unknown').")
    f = st.file_uploader("Upload a clear photo of one enrolled staff member",
                         type=["jpg", "jpeg", "png"])

    if f:
        try:
            img = Image.open(f).convert("RGB")
        except OSError as exc:
            st.error(f"Could not read uploaded photo: {exc}")
        else:
            boxes, faces = fe.detect_faces(img)
            if len(boxes) != 1:
                st.warning("Upload a photo containing exactly one clearly visible face.")
            else:
                idx, sims = fe.nearest(fe.embed_tensors(faces), G)
                best_index, similarity = int(idx[0]), float(sims[0])
                matched_pid = owners[best_index] if similarity >= thr else None
                box = boxes[0]
                draw = ImageDraw.Draw(img)
                draw.rectangle(list(box), outline="lime" if matched_pid else "red", width=4)
                st.image(img, alt="Uploaded staff photo checked for attendance", width="stretch")

                if matched_pid:
                    st.success(f"Possible match: {gallery[matched_pid]['name']} "
                               f"({similarity:.3f} similarity)")
                    enrolled_photo = staff_photo_path(matched_pid)
                    if enrolled_photo.exists():
                        st.image(enrolled_photo, caption="Saved enrolment photo",
                                 alt=f"Enrolment photo for {gallery[matched_pid]['name']}",
                                 width=200)
                    else:
                        st.info("This staff member was enrolled before photo storage was added. "
                                "Re-enrol them to save a profile photo.")
                    st.caption("Confirm the match before marking this person present.")
                    if st.button("Confirm match and mark present", type="primary"):
                        save_attendance_statuses(
                            [{
                                "person_id": matched_pid,
                                "name": gallery[matched_pid]["name"],
                                "status": "Present",
                                "similarity": similarity,
                            }],
                            today,
                        )
                        st.success(f"Marked {gallery[matched_pid]['name']} present.")
                else:
                    st.warning(f"No enrolled staff match found ({similarity:.3f} similarity).")

# ----------------------------- REPORTS ---------------------------------------
else:
    st.header("Attendance report")
    log = load_log()
    week_reference = st.date_input(
        "Choose a date in the week",
        value=dt.date.today(),
        max_value=dt.date.today(),
    )
    monday = week_reference - dt.timedelta(days=week_reference.weekday())
    weekdays = [monday + dt.timedelta(days=offset) for offset in range(5)]
    available_days = [weekday for weekday in weekdays if weekday <= dt.date.today()]
    day = st.selectbox(
        "Attendance day (Monday-Friday)",
        available_days,
        index=min(week_reference.weekday(), len(available_days) - 1),
        format_func=lambda value: value.strftime("%A, %B %d"),
    )
    st.caption(f"Weekdays: {weekdays[0]:%b %d} - {weekdays[-1]:%b %d, %Y}")

    recorded = log[log["date"] == str(day)]
    recorded_status = dict(zip(recorded["person_id"], recorded["status"]))
    roster = pd.DataFrame([
        {
            "person_id": person_id,
            "name": details["name"],
            "status": recorded_status.get(person_id, "Absent"),
        }
        for person_id, details in gallery.items()
    ])
    if roster.empty:
        st.info("Enrol staff to see the weekday attendance roster.")
        roster_statuses = pd.Series(dtype=str)
    else:
        log_version = LOG_FILE.stat().st_mtime_ns if LOG_FILE.exists() else 0
        edited_roster = st.data_editor(
            roster,
            column_config={
                "person_id": st.column_config.TextColumn("Staff ID"),
                "name": st.column_config.TextColumn("Name"),
                "status": st.column_config.SelectboxColumn(
                    "Status", options=["Present", "Absent"], required=True
                ),
            },
            disabled=["person_id", "name"],
            hide_index=True,
            key=f"attendance_roster_{day}_{log_version}",
            alt=f"Attendance status for {day:%A, %B %d, %Y}",
        )
        roster_statuses = edited_roster["status"]
        if st.button("Save attendance", type="primary"):
            records = edited_roster.to_dict("records")
            save_attendance_statuses(records, day)
            st.success(f"Saved attendance for {day:%A, %B %d, %Y}.")

    present_count = int((roster_statuses == "Present").sum())
    absent_count = len(roster_statuses) - present_count
    m1, m2, m3 = st.columns(3)
    m1.metric("Present", present_count)
    m2.metric("Absent", absent_count)
    m3.metric("Rate", f"{100 * present_count / max(len(gallery), 1):.0f}%")
    st.download_button("Download full log (CSV)", log.to_csv(index=False),
                       "attendance.csv")
