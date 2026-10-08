"""
face_engine.py - shared AI core used by train_evaluate.py and app.py

Pipeline:  image -> MTCNN (detect + align) -> InceptionResnetV1 (FaceNet, pretrained
on VGGFace2) -> 512-d embedding -> L2 normalise -> cosine similarity vs. gallery
"""
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _detector():
    # keep_all=True -> detect every face (needed for group photos)
    return MTCNN(image_size=160, margin=20, keep_all=True,
                 post_process=True, device=DEVICE)


@lru_cache(maxsize=1)
def _cnn():
    # Pretrained CNN. Weights are downloaded automatically on first run.
    return InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def _l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def embed_tensors(tensors, batch=64):
    """tensors: (N,3,160,160) standardised -> (N,512) L2-normalised embeddings."""
    out = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch):
            out.append(_cnn()(tensors[i:i + batch].to(DEVICE)).cpu().numpy())
    return _l2(np.concatenate(out))


def arrays_to_tensors(imgs):
    """Already-cropped face images (uint8 HxWx3) -> standardised tensors.
    Same standardisation MTCNN applies: (x - 127.5) / 128."""
    ts = []
    for a in imgs:
        im = Image.fromarray(a).resize((160, 160))
        x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)
        ts.append((x - 127.5) / 128.0)
    return torch.stack(ts)


def embed_arrays(imgs):
    return embed_tensors(arrays_to_tensors(imgs))


def detect_faces(pil_img, min_prob=0.90):
    """Return (boxes[n,4], face_tensors[n,3,160,160]); (empty, None) if no face."""
    boxes, probs = _detector().detect(pil_img)
    if boxes is None:
        return np.empty((0, 4)), None
    keep = np.array([p is not None and p >= min_prob for p in probs])
    boxes = boxes[keep]
    if len(boxes) == 0:
        return np.empty((0, 4)), None
    faces = _detector().extract(pil_img, boxes, None)
    return boxes, faces


def nearest(query, gallery):
    """Cosine similarity (embeddings are unit length -> dot product).
    Returns index of best gallery row and its similarity, for each query."""
    sims = query @ gallery.T
    idx = sims.argmax(axis=1)
    return idx, sims[np.arange(len(query)), idx]
