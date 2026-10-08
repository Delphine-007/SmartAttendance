# Smart Face-Recognition Attendance System

**Module:** ITLPA701 – Python and Fundamentals of AI
**Problem domain:** Image recognition (face recognition) for staff attendance
**AI approach:** Deep Learning – pretrained CNN (FaceNet / InceptionResnetV1) + MTCNN detector

## 1. Problem
Manual roll-calls waste time, allow "buddy punching" and are hard to report on.
A manager needs a quick way to record who is present and see who is absent.

## 2. Architecture

```
 Staff photo upload
        |
        v
 [MTCNN]  detect + align one face (prob >= 0.90)
        |
        v
 [InceptionResnetV1 - CNN, VGGFace2 pretrained]  -> 512-d embedding (L2-normalised)
        |
        v
 [Cosine similarity vs. gallery of enrolled staff]
        |-- similarity >= threshold --> staff name
        |-- similarity <  threshold --> "Unknown"
        v
 Manager confirmation -> weekday roster -> Present / Absent report (Streamlit)

```

## 3. Why this approach (justification)
- Only ~5 photos per person exist, so training a CNN from scratch would overfit.
  Transfer learning from a model trained on 3.3M faces (VGGFace2) is far better.
- Embedding + similarity means **new staff can be enrolled instantly with no retraining**.
- A threshold lets the system say **"Unknown"** for strangers (open-set recognition),
  which a plain softmax classifier cannot do.

## 4. Important parameters
| Parameter | Value | Reason |
|---|---|---|
| Face size / margin | 160 px / 20 px | Input size expected by FaceNet |
| Detection probability | ≥ 0.90 | Ignore blurry/false detections |
| Embedding | 512-d, L2-normalised | Cosine similarity = dot product |
| Enrol photos / person | 5 | Covers angle and lighting variety |
| Similarity threshold | tuned (see `models/config.json`) | Balances false accepts vs. false rejects |

## 5. How to run
```bash
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt

venv\Scripts\python.exe train_evaluate.py      # data prep + evaluation + threshold tuning (needs internet 1st time)
venv\Scripts\python.exe -m streamlit run app.py
```

## 6. Evaluation
`train_evaluate.py` uses LFW (scikit-learn). It enrols 5 photos per person, tests on their
other photos, and also tests **strangers** who must be rejected.
Metrics: accuracy, precision, recall, F1 (macro), confusion matrix, stranger-rejection rate,
and time per image. The threshold is swept from 0.30 to 0.90 and the value with best macro-F1
is saved. Outputs: `models/threshold_sweep.csv`, `classification_report.txt`,
`confusion_matrix.png`, `threshold_curve.png`, `config.json`.

*(Write your own interpretation here after running: e.g. "At threshold 0.xx, accuracy = xx%, strangers rejected = xx%.
Lower thresholds accept strangers; higher thresholds reject real staff.")*

## 7. Reproducibility
Saved: `models/config.json` (model, threshold, seed), `data/gallery.pkl` (enrolled embeddings),
`data/attendance.csv` (log). Pretrained weights are re-downloaded automatically.

## 8. Responsible use
| Risk / limitation | How to reduce it |
|---|---|
| **Privacy** – face data is sensitive | Explicit consent checkbox; store embeddings and enrolment photos locally; delete function; restrict access to the PC; follow Rwanda's personal-data protection law (Law No. 058/2021) |
| **Spoofing** – a printed photo can fool it | Manager supervises; add liveness / blink detection in future |
| **Bias** – accuracy can differ by skin tone, lighting, age | Enrol under varied lighting; test on your own staff; keep manual fallback |
| **Wrong match** | Threshold tuning + manager must confirm before recording |

## 9. Rubric mapping
| Rubric item | Where |
|---|---|
| Environment configured | `requirements.txt`, section 5 |
| Data acquired | LFW via scikit-learn (`train_evaluate.py` §1) |
| Data pre-processed | class balancing, resize 160×160, standardisation (§2-3) |
| Features engineered | 512-d FaceNet embeddings, L2-norm |
| Approach selected & justified | sections 3 |
| Parameters explained | section 4 |
| Application implemented & tested | `face_engine.py`, `app.py` |
| Metrics selected / implemented / interpreted | `train_evaluate.py` §4, section 6 |
| Parameters adjusted from results | threshold sweep |
| Saved for reproducibility | section 7 |
| Deployed via web interface | Streamlit `app.py` |
| Responsible use | section 8 |
