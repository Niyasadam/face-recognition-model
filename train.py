"""
Builds a face embedding "gallery" using a pretrained FaceNet model.

Run after collecting data with collect.py:

    python train.py

Unlike LBPH, there's no classical "training" step. Each image is run
through MTCNN (face detection + alignment) and InceptionResnetV1
(embedding), and the resulting 512-d embeddings are stored alongside
subject labels in face_embeddings.npz.
"""

import argparse
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
MAX_EMBEDDINGS_PER_SUBJECT = 150


def save_checkpoint(embeddings, labels, mapping, processed):
    np.savez(
        EMBEDDINGS_PATH,
        embeddings=np.array(embeddings, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        mapping=np.array(list(mapping.items()), dtype=object),
        processed=np.array(list(processed), dtype=object),
    )


def load_embeddings(full=False):
    if not FACE_DATA_DIR.exists():
        return [], [], {}, set()

    subject_dirs = sorted(d.name for d in FACE_DATA_DIR.iterdir() if d.is_dir())
    if not subject_dirs:
        return [], [], {}, set()

    embeddings, labels, mapping = [], [], {}
    processed = set()

    # If incremental, load existing data
    if not full and EMBEDDINGS_PATH.exists():
        try:
            data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
            embeddings = list(data["embeddings"])
            labels = list(data["labels"])
            mapping = {int(k): str(v) for k, v in data["mapping"]}
            if "processed" in data:
                processed = set(data["processed"])
            print(f"Loaded existing gallery with {len(embeddings)} embeddings for {len(mapping)} subjects.")
        except Exception as e:
            print(f"Could not load existing gallery ({e}). Starting fresh.")
            embeddings, labels, mapping, processed = [], [], {}, set()

    # Filter/cleanup to only keep active subjects (handle deleted folders)
    active_subjects = set(subject_dirs)
    new_mapping = {}
    name_to_id = {}
    next_label_id = 0

    for label_id, name in mapping.items():
        if name in active_subjects:
            new_mapping[label_id] = name
            name_to_id[name] = label_id
            if label_id >= next_label_id:
                next_label_id = label_id + 1

    for name in subject_dirs:
        if name not in name_to_id:
            new_mapping[next_label_id] = name
            name_to_id[name] = next_label_id
            next_label_id += 1

    # Filter existing arrays
    filtered_embeddings = []
    filtered_labels = []
    for emb, lbl in zip(embeddings, labels):
        if lbl in new_mapping:
            filtered_embeddings.append(emb)
            filtered_labels.append(lbl)

    filtered_processed = set()
    for p_path in processed:
        parts = p_path.split("/", 1)
        if len(parts) == 2 and parts[0] in active_subjects:
            filtered_processed.add(p_path)

    embeddings = filtered_embeddings
    labels = filtered_labels
    mapping = new_mapping
    processed = filtered_processed

    total_images = 0
    skipped = 0

    for name in subject_dirs:
        folder = FACE_DATA_DIR / name
        files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXT]

        if len(files) < MIN_IMAGES_PER_SUBJECT:
            print(f"WARNING: '{name}' has only {len(files)} images "
                  f"(recommend {MIN_IMAGES_PER_SUBJECT}+ for reliable recognition)")

        label_id = name_to_id[name]
        current_count = labels.count(label_id)

        # Skip files already processed
        new_files = []
        for f in files:
            rel_path = f"{name}/{f.name}"
            if rel_path in processed:
                total_images += 1
            else:
                new_files.append((f, rel_path))

        if not new_files:
            print(f"[{name}] already up-to-date ({current_count} embeddings in gallery).")
            continue

        print(f"[{name}] found {len(new_files)} new images (current embeddings: {current_count}). Processing...")
        new_embeddings = []

        for fp, rel_path in new_files:
            total_images += 1

            if current_count + len(new_embeddings) >= MAX_EMBEDDINGS_PER_SUBJECT:
                print(f"   -> Reached MAX_EMBEDDINGS_PER_SUBJECT ({MAX_EMBEDDINGS_PER_SUBJECT}) for '{name}'. Skipping remaining new images.")
                break

            try:
                img = Image.open(fp).convert("RGB")
            except Exception:
                skipped += 1
                continue

            emb, prob = embed_largest_face(img)
            if emb is None:
                skipped += 1
                processed.add(rel_path)
                continue

            new_embeddings.append(emb)
            processed.add(rel_path)

        if new_embeddings:
            for emb in new_embeddings:
                embeddings.append(emb)
                labels.append(label_id)
            print(f"   -> Added {len(new_embeddings)} new embeddings to gallery")
            
            save_checkpoint(embeddings, labels, mapping, processed)

    print()
    print(f"Total source images scanned: {total_images}")
    print(f"Skipped (no usable face)   : {skipped}")
    return embeddings, labels, mapping, processed


def find_best_threshold(embeddings, labels, mapping):
    """
    Optimized Threshold Evaluator using Vectorized Matrix Math.
    Runs matching across all elements instantly without internal loops.
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

    gallery_emb = embeddings[gallery_idx]  # Shape: (G, 512)
    gallery_lbl = labels[gallery_idx]      # Shape: (G,)
    query_emb = embeddings[query_idx]      # Shape: (Q, 512)
    query_lbl = labels[query_idx]          # Shape: (Q,)

    print()
    print("Evaluating recognition accuracy across distance thresholds...")
    
    # ─── OPTIMIZED MATRIX MATH SELECTION ───
    # Compute the full pairwise Euclidean distance matrix using vector geometry:
    # ||A - B|| = sqrt(A^2 + B^2 - 2AB)
    q_sq = np.sum(query_emb ** 2, axis=1, keepdims=True)  # (Q, 1)
    g_sq = np.sum(gallery_emb ** 2, axis=1, keepdims=True).T  # (1, G)
    cross_term = dot_prod = np.dot(query_emb, gallery_emb.T)  # (Q, G)
    
    # Secure matrix clipping to eliminate potential floating point noise errors below 0.0
    dist_matrix = np.sqrt(np.clip(q_sq + g_sq - 2 * cross_term, 0.0, None)) # Final Matrix Shape: (Q, G)
    
    # Find the single closest vector coordinate inside the training gallery for each test face
    closest_gallery_indices = np.argmin(dist_matrix, axis=1)
    closest_distances = dist_matrix[np.arange(len(query_idx)), closest_gallery_indices]
    predicted_labels = gallery_lbl[closest_gallery_indices]
    
    # Sweep thresholds instantly using masking operations
    results = {}
    threshold_range = np.arange(0.40, 1.55, 0.05)
    
    for thresh in threshold_range:
        thresh = round(float(thresh), 2)
        
        # A match is valid if it is under the threshold and the label matches perfectly
        is_recognized = closest_distances <= thresh
        is_correct = is_recognized & (predicted_labels == query_lbl)
        
        correct_count = int(np.sum(is_correct))
        recognized_count = int(np.sum(is_recognized))
        
        accuracy = correct_count / len(query_emb)
        results[thresh] = (accuracy, recognized_count)

    best_thresh = max(results, key=lambda t: results[t][0])

    print()
    print(f"{'Threshold':>10} | {'Accuracy':>9} | {'Recognized':>10} / {len(query_emb)}")
    print("-" * 45)
    for thresh, (acc, recognized) in results.items():
        marker = "  <-- best" if thresh == best_thresh else ""
        print(f"{thresh:>10} | {acc*100:>8.1f}% | {recognized:>10}{marker}")

    print()
    print(f"Recommended FACE_DISTANCE_THRESHOLD for app.py: {best_thresh}")
    print("(Lower = stricter matching, more 'Unknown' results; higher = more lenient.)")
    return best_thresh


def main():
    parser = argparse.ArgumentParser(description="Build or update face embedding gallery")
    parser.add_argument("--full", action="store_true", help="Rebuild embedding gallery from scratch")
    args = parser.parse_args()

    print("=" * 60)
    print("Building face embedding gallery (FaceNet / InceptionResnetV1)")
    print("=" * 60)
    print()

    embeddings, labels, mapping, processed = load_embeddings(full=args.full)

    if not embeddings:
        print("\nNo valid training data found in face_data/.")
        return

    print()
    print(f"Subjects: {len(mapping)}")
    
    find_best_threshold(embeddings, labels, mapping)
    save_checkpoint(embeddings, labels, mapping, processed)

    print()
    print(f"Embedding gallery saved -> {EMBEDDINGS_PATH}")
    print("Done. Run `streamlit run app.py` to use the model.")


if __name__ == "__main__":
    main()