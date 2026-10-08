"""
train_evaluate.py
Steps covered: data acquisition, preprocessing, feature engineering (embeddings),
evaluation, parameter tuning (similarity threshold) and saving the configuration.

Data source : LFW (Labeled Faces in the Wild) via scikit-learn  (approved source)
Simulation  : some people are ENROLLED (5 photos each = gallery),
              their other photos are the attendance TEST images,
              other people are STRANGERS who must be rejected as UNKNOWN.

Run:  python train_evaluate.py
"""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import face_engine as fe

# ----------------------------- settings --------------------------------------
SEED = 42
MIN_FACES = 40        # keep people with >= 40 photos
MAX_PER_PERSON = 40   # cap -> balanced classes (Bush has 500+ photos otherwise)
N_STRANGERS = 5       # people never enrolled (must be rejected)
N_ENROL = 5           # photos used to enrol each person
OUT = Path("models")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)

# ----------------------- 1. data acquisition ---------------------------------
print("Downloading / loading LFW ...")
lfw = fetch_lfw_people(min_faces_per_person=MIN_FACES, color=True, resize=1.0)
images = (lfw.images * 255).clip(0, 255).astype(np.uint8)   # sklearn gives 0-1 floats
y_all, names = lfw.target, lfw.target_names
print(f"Loaded {len(images)} images of {len(names)} people, shape {images.shape[1:]}")

# ----------------------- 2. preprocessing ------------------------------------
# balance classes
keep = []
for c in np.unique(y_all):
    idx = np.where(y_all == c)[0]
    rng.shuffle(idx)
    keep.append(idx[:MAX_PER_PERSON])
per_person = {c: idx for c, idx in zip(np.unique(y_all), keep)}

classes = rng.permutation(np.unique(y_all))
strangers, enrolled = classes[:N_STRANGERS], classes[N_STRANGERS:]

gal_idx, gal_lab, test_idx, test_lab = [], [], [], []
for c in enrolled:
    idx = per_person[c]
    gal_idx += list(idx[:N_ENROL]);  gal_lab += [c] * N_ENROL
    test_idx += list(idx[N_ENROL:]); test_lab += [c] * (len(idx) - N_ENROL)
for c in strangers:                       # label -1 == UNKNOWN
    idx = per_person[c]
    test_idx += list(idx); test_lab += [-1] * len(idx)

gal_lab, y_true = np.array(gal_lab), np.array(test_lab)
print(f"Enrolled people: {len(enrolled)} | gallery images: {len(gal_idx)} | "
      f"test images: {len(test_idx)} (strangers: {(y_true == -1).sum()})")

# ----------------------- 3. feature engineering ------------------------------
# LFW images are already face crops -> resize to 160x160 + standardise -> FaceNet.
print("Computing FaceNet embeddings ...")
E_gal = fe.embed_arrays(images[gal_idx])
t0 = time.perf_counter()
E_test = fe.embed_arrays(images[test_idx])
ms_per_img = 1000 * (time.perf_counter() - t0) / len(test_idx)
print(f"Embedding dimension: {E_test.shape[1]} | {ms_per_img:.1f} ms / image on {fe.DEVICE}")

# ----------------------- 4. evaluation + tuning ------------------------------
best_idx, best_sim = fe.nearest(E_test, E_gal)
pred_class = gal_lab[best_idx]
labels = np.unique(y_true)
rows = []
for thr in np.arange(0.30, 0.91, 0.05):
    pred = np.where(best_sim >= thr, pred_class, -1)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=labels, average="macro", zero_division=0)
    known = y_true != -1
    rows.append(dict(threshold=round(thr, 2),
                     accuracy=accuracy_score(y_true, pred),
                     precision_macro=p, recall_macro=r, f1_macro=f1,
                     known_person_accuracy=(pred[known] == y_true[known]).mean(),
                     stranger_rejection_rate=(pred[~known] == -1).mean()))
sweep = pd.DataFrame(rows)
sweep.to_csv(OUT / "threshold_sweep.csv", index=False)
print("\nThreshold tuning:\n", sweep.round(3).to_string(index=False))

best = sweep.loc[sweep.f1_macro.idxmax()]
thr = float(best.threshold)
print(f"\n>>> Best threshold = {thr} (macro F1 = {best.f1_macro:.3f})")

pred = np.where(best_sim >= thr, pred_class, -1)
label_names = [names[l] if l != -1 else "UNKNOWN" for l in labels]
report = classification_report(y_true, pred, labels=labels,
                               target_names=label_names, zero_division=0)
print(report)
(OUT / "classification_report.txt").write_text(report)

# confusion matrix
cm = confusion_matrix(y_true, pred, labels=labels)
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
    ax=ax, xticks_rotation=90, colorbar=False, cmap="Blues")
ax.set_title(f"Confusion matrix (threshold={thr})")
plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=130); plt.close()

# threshold curves
fig, ax = plt.subplots(figsize=(7, 4))
for col in ["known_person_accuracy", "stranger_rejection_rate", "f1_macro"]:
    ax.plot(sweep.threshold, sweep[col], marker="o", label=col)
ax.axvline(thr, ls="--", c="gray"); ax.set_xlabel("cosine-similarity threshold")
ax.legend(); ax.grid(alpha=.3); plt.tight_layout()
plt.savefig(OUT / "threshold_curve.png", dpi=130); plt.close()

# ----------------------- 5. save configuration (reproducibility) -------------
cfg = dict(model="InceptionResnetV1 (VGGFace2 pretrained)", detector="MTCNN",
           embedding_dim=int(E_test.shape[1]), similarity="cosine",
           threshold=thr, min_detection_prob=0.90, enrol_photos_per_person=N_ENROL,
           seed=SEED, ms_per_image=round(ms_per_img, 1),
           overall_accuracy=round(float(best.accuracy), 4),
           macro_f1=round(float(best.f1_macro), 4))
(OUT / "config.json").write_text(json.dumps(cfg, indent=2))
print("Saved models/config.json, confusion_matrix.png, threshold_curve.png")
