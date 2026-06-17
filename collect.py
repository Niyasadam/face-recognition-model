"""
Data collection CLI.

Two modes:

1. Import a celebrity dataset zip (handles nested folder structures like
   actors/personname/image.jpg -- the DIRECT PARENT folder name becomes
   the subject's identity):

       python collect.py zip dataset1.zip
       python collect.py zip dataset2.zip

2. Capture face samples from your webcam for a new/existing subject:

       python collect.py webcam "Your Name" --count 40

All images end up under face_data/<subject_name>/, ready for train.py.
"""

import argparse
import shutil
import sys
import subprocess
import zipfile
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).parent.resolve()
FACE_DATA_DIR = BASE_DIR / "face_data"

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def import_zip(zip_path: str) -> None:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        print(f"Zip file not found: {zip_path}")
        return

    tmp_dir = BASE_DIR / "_tmp_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir()

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp_dir)

    FACE_DATA_DIR.mkdir(exist_ok=True)

    imported = 0
    skipped = 0
    for img_path in tmp_dir.rglob("*"):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in VALID_EXT:
            skipped += 1
            continue

        subject = img_path.parent.name.strip()
        if not subject:
            skipped += 1
            continue

        dest_dir = FACE_DATA_DIR / subject
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest = dest_dir / img_path.name
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
            counter += 1

        shutil.copy2(img_path, dest)
        imported += 1

    shutil.rmtree(tmp_dir)
    print(f"Imported {imported} images, skipped {skipped} non-image files.")
    print(f"Subjects now in face_data/: {sorted(d.name for d in FACE_DATA_DIR.iterdir() if d.is_dir())}")


def collect_webcam(name: str, num_images: int, cam_index: int) -> None:
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    dest_dir = FACE_DATA_DIR / name.strip()
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_count = len(list(dest_dir.glob("*")))

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Could not open camera index {cam_index}.")
        return

    print("Press SPACE to capture a sample, ESC to quit early.")
    saved = 0
    stem = name.strip().replace(" ", "_").lower()

    while saved < num_images:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        display = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 150), 2)

        status = f"Captured {saved}/{num_images}  |  SPACE = capture, ESC = quit"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
        cv2.imshow("Face Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and len(faces) > 0:  # SPACE
            idx = existing_count + saved + 1
            fname = dest_dir / f"{stem}_{idx:04d}.png"
            # Save the FULL frame (not a tight crop) -- the FaceNet
            # pipeline's MTCNN detector aligns/crops with its own margin
            # during embedding extraction, so extra context around the
            # face helps rather than hurts.
            cv2.imwrite(str(fname), frame)
            saved += 1
            print(f"Saved {fname.name} ({saved}/{num_images})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Collected {saved} new images for '{name}'.")


def main():
    parser = argparse.ArgumentParser(description="Face dataset collection tool")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_zip = sub.add_parser("zip", help="Import images from a celebrity dataset zip")
    p_zip.add_argument("path", help="Path to the zip file")
    p_zip.add_argument("--no-train", action="store_true", help="Skip automatic gallery training")

    p_cam = sub.add_parser("webcam", help="Capture face images from your webcam")
    p_cam.add_argument("name", help="Subject name (folder will be created under face_data/)")
    p_cam.add_argument("--count", type=int, default=30, help="Number of samples to capture (default 30)")
    p_cam.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p_cam.add_argument("--no-train", action="store_true", help="Skip automatic gallery training")

    args = parser.parse_args()

    no_train = False
    if args.mode == "zip":
        import_zip(args.path)
        no_train = args.no_train
    elif args.mode == "webcam":
        collect_webcam(args.name, args.count, args.camera)
        no_train = args.no_train

    if not no_train:
        print("\nTriggering automatic gallery training...")
        train_script = Path(__file__).parent / "train.py"
        try:
            subprocess.run([sys.executable, str(train_script)], check=True)
        except Exception as e:
            print(f"Failed to run automatic training: {e}")


if __name__ == "__main__":
    main()
    