"""
Builds a face embedding "gallery" using a pretrained FaceNet model.

Run after collecting data with collect.py:

    python train.py

Unlike LBPH, there's no classical "training" step. Each image is run
through MTCNN (face detection + alignment) and InceptionResnetV1
(embedding), and the resulting 512-d embeddings are stored alongside
subject labels in face_embeddings.npz.

Recognition at inference time (app.py) is nearest-neighbor search: a
new face's embedding is compared against every stored embedding, and
the closest match (if within FACE_DISTANCE_THRESHOLD) determines the
identity.

Because recognition is nearest-neighbor rather than a trained
classifier, having very different numbers of images per subject
(e.g. 13 vs 300) is far less of a problem than it was with LBPH --
each embedding is just an extra reference point, it doesn't bias a
decision boundary.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from face_engine import embed_largest_face

BASE_DIR = Path(__file__).parent.resolve()
FACE_DATA_DIR = BASE_DIR / "face_data"
EMBEDDINGS_PATH = BASE_DIR / "face_embeddings.npz"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
MIN_IMAGES_PER_SUBJECT = 5

# Cap how many embeddings any single subject contributes to the gallery.
# Keeps the gallery (and per-query search time) manageable for subjects
# with very large image counts, without hurting accuracy.
MAX_EMBEDDINGS_PER_SUBJECT = 150


def load_embeddings():
    if not FACE_DATA_DIR.exists():
        return [], [], {}

    subject_dirs = sorted(d.name for d in FACE_DATA_DIR.iterdir() if d.is_dir())
    if not subject_dirs:
        return [], [], {}

    embeddings, labels, mapping = [], [], {}
    next_label_id = 0
    total_images = 0
    skipped = 0

    for name in subject_dirs:
        folder = FACE_DATA_DIR / name
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT]

        if len(files) < MIN_IMAGES_PER_SUBJECT:
            print(f"WARNING: '{name}' has only {len(files)} images "
                  f"(recommend {MIN_IMAGES_PER_SUBJECT}+ for reliable recognition)")

        print(f"[{name}] processing {len(files)} images...")
        subject_embeddings = []

        for fp in files:
            total_images += 1
            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                skipped += 1
                continue

            emb, prob = embed_largest_face(img)
            if emb is None:
                skipped += 1
                continue

            subject_embeddings.append(emb)

        if not subject_embeddings:
            print(f"   -> SKIPPING '{name}': no usable faces detected in any image")
            continue

        if len(subject_embeddings) > MAX_EMBEDDINGS_PER_SUBJECT:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(subject_embeddings), size=MAX_EMBEDDINGS_PER_SUBJECT, replace=False)
            subject_embeddings = [subject_embeddings[i] for i in idx]

        label_id = next_label_id
        mapping[label_id] = name
        next_label_id += 1

        for emb in subject_embeddings:
            embeddings.append(emb)
            labels.append(label_id)

        print(f"   -> {len(subject_embeddings)} embeddings added to gallery")

    print()
    print(f"Total source images scanned: {total_images}")
    print(f"Skipped (no usable face)   : {skipped}")
    return embeddings, labels, mapping


def find_best_threshold(embeddings, labels, mapping):
    """
    Holds out ~30% of each subject's embeddings as queries, runs 1-NN
    against the remaining gallery, and sweeps L2-distance thresholds to
    find the best cutoff for classifying a face as "Unknown".
    """
    rng = np.random.default_rng(42)
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    gallery_idx, query_idx = [], []
    for label_id in mapping:
        idx = np.where(labels == label_id)[0]
        rng.shuffle(idx)
        n_query = max(1, int(len(idx) * 0.3)) if len(idx) > 1 else 0
        query_idx.extend(idx[:n_query])
        gallery_idx.extend(idx[n_query:])

    if not query_idx or not gallery_idx:
        print("\nNot enough embeddings per subject to evaluate thresholds; skipping.")
        return None

    gallery_emb = embeddings[gallery_idx]
    gallery_lbl = labels[gallery_idx]
    query_emb = embeddings[query_idx]
    query_lbl = labels[query_idx]

    print()
    print("Evaluating recognition accuracy across distance thresholds...")
    results = {}
    for thresh in np.arange(0.40, 1.55, 0.05):
        thresh = round(float(thresh), 2)
        correct = 0
        recognized = 0
        for q_emb, true_label in zip(query_emb, query_lbl):
            dists = np.linalg.norm(gallery_emb - q_emb, axis=1)
            min_idx = int(np.argmin(dists))
            min_dist = float(dists[min_idx])
            if min_dist <= thresh:
                recognized += 1
                if gallery_lbl[min_idx] == true_label:
                    correct += 1
        accuracy = correct / len(query_emb)
        results[thresh] = (accuracy, recognized)

    best_thresh = max(results, key=lambda t: results[t][0])

    print()
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Recognized':>10} / {len(query_emb)}")
    print("-" * 45)
    for thresh, (acc, recognized) in results.items():
        marker = "  <-- best" if thresh == best_thresh else ""
        print(f"{thresh:>10} | {acc*100:>8.1f}% | {recognized:>10}{marker}")

    print()
    print(f"Recommended FACE_DISTANCE_THRESHOLD for app.py: {best_thresh}")
    print("(Lower = stricter matching, more 'Unknown' results;")
    print(" higher = more lenient, more chance of false matches.)")
    return best_thresh


def main():
    print("=" * 60)
    print("Building face embedding gallery (FaceNet / InceptionResnetV1)")
    print("=" * 60)
    print("Note: first run downloads pretrained weights (~110MB), needs internet.")
    print()

    embeddings, labels, mapping = load_embeddings()

    if not embeddings:
        print("\nNo valid training data found in face_data/.")
        print("Use collect.py to import a dataset zip or capture webcam samples first.")
        return

    print()
    print(f"Subjects: {len(mapping)}")
    for label_id, name in mapping.items():
        print(f"   [{label_id}] {name}: {labels.count(label_id)} embeddings")

    find_best_threshold(embeddings, labels, mapping)

    np.savez(
        EMBEDDINGS_PATH,
        embeddings=np.array(embeddings, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        mapping=np.array(list(mapping.items()), dtype=object),
    )

    print()
    print(f"Embedding gallery saved -> {EMBEDDINGS_PATH}")
    print("Done. Run `streamlit run app.py` to use the model.")


if __name__ == "__main__":
    main()