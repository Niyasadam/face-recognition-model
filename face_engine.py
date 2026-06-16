"""
Shared face detection + embedding utilities using a pretrained FaceNet
model (InceptionResnetV1 trained on VGGFace2) via facenet-pytorch.

This replaces the previous OpenCV/LBPH-based face_utils.py. Recognition
now works by:
  1. MTCNN detects and aligns face crops (much better than Haar cascades
     across pose/lighting/angle)
  2. InceptionResnetV1 converts each aligned face into a 512-d,
     L2-normalized embedding vector
  3. Recognition = nearest-neighbor search in embedding space (see
     train.py for gallery building and app.py for inference)

First run will download the pretrained weights (~110MB) to
~/.cache/torch -- this requires an internet connection once.
"""

import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FACE_SIZE = 160
DETECTION_PROB_THRESHOLD = 0.90  # ignore low-confidence "face" detections

# Detector configured to find ALL faces in an image (used for both
# training and live/photo recognition)
_mtcnn = MTCNN(keep_all=True, device=DEVICE)

# Pretrained embedding model (downloads weights on first use)
_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def detect_faces(pil_img):
    """
    Detect faces in a PIL RGB image.
    Returns a list of (box, prob) tuples, box = [x1, y1, x2, y2],
    filtered to detections above DETECTION_PROB_THRESHOLD.
    """
    boxes, probs = _mtcnn.detect(pil_img)
    results = []
    if boxes is None:
        return results
    for box, prob in zip(boxes, probs):
        if prob is not None and prob >= DETECTION_PROB_THRESHOLD:
            results.append((box, float(prob)))
    return results


def embed_face(pil_img, box):
    """
    Crop+align the face at `box` and return its 512-d L2-normalized
    FaceNet embedding as a numpy array.
    """
    face_tensor = extract_face(pil_img, box, image_size=FACE_SIZE)
    face_tensor = fixed_image_standardization(face_tensor)
    with torch.no_grad():
        emb = _resnet(face_tensor.unsqueeze(0).to(DEVICE))
    return emb.cpu().numpy()[0]


def embed_largest_face(pil_img):
    """
    Convenience for training: detect all faces in an image and embed
    the largest one (by box area). Returns (embedding, prob), or
    (None, None) if no face was detected.
    """
    faces = detect_faces(pil_img)
    if not faces:
        return None, None

    def area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    box, prob = max(faces, key=lambda f: area(f[0]))
    return embed_face(pil_img, box), prob
