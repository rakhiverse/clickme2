import os

# ============================================================
# LOW MEMORY / LOW CPU SETTINGS
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import cv2
import numpy as np
import insightface


# ============================================================
# GLOBAL FACE APP
# ============================================================

app = None


# ============================================================
# LOAD INSIGHTFACE MODEL
# ============================================================

def get_app():
    global app

    if app is not None:
        return app

    print("[FACE] Loading InsightFace buffalo_s model...")

    try:
        app = insightface.app.FaceAnalysis(
            name="buffalo_s",
            allowed_modules=[
                "detection",
                "recognition"
            ]
        )

        # CPU only.
        # 320x320 keeps RAM and CPU usage lower.
        app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        print(
            "[FACE] InsightFace buffalo_s "
            "loaded successfully."
        )

        return app

    except Exception as e:
        app = None

        print(
            "[FACE ERROR] Could not load "
            f"InsightFace: {repr(e)}"
        )

        raise


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path: str):

    try:
        img = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if img is None:
            print(
                f"[FACE] Image could not be loaded: "
                f"{image_path}"
            )
            return None

        return img

    except Exception as e:
        print(
            f"[FACE] Image loading error: "
            f"{repr(e)}"
        )
        return None


# ============================================================
# RESIZE LARGE IMAGE
# ============================================================

def resize_image(img):

    try:
        height, width = img.shape[:2]

        max_dimension = max(
            height,
            width
        )

        # Keep already-small images unchanged.
        if max_dimension <= 1200:
            return img

        scale = 1200.0 / float(max_dimension)

        new_width = max(
            1,
            int(width * scale)
        )

        new_height = max(
            1,
            int(height * scale)
        )

        resized = cv2.resize(
            img,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

        print(
            f"[FACE] Image resized: "
            f"{width}x{height} -> "
            f"{new_width}x{new_height}"
        )

        return resized

    except Exception as e:
        print(
            f"[FACE] Resize error: "
            f"{repr(e)}"
        )

        return img


# ============================================================
# GET FACE EMBEDDINGS
# ============================================================

def get_face_embeddings(image_path: str):

    try:

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = load_image(
            image_path
        )

        if img is None:
            return []

        # ----------------------------------------------------
        # Resize image
        # ----------------------------------------------------

        img = resize_image(
            img
        )

        # ----------------------------------------------------
        # Load InsightFace
        # ----------------------------------------------------

        face_app = get_app()

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_app.get(
            img
        )

        if not faces:
            print(
                f"[FACE] No face detected: "
                f"{image_path}"
            )
            return []

        results = []

        # ----------------------------------------------------
        # Process detected faces
        # ----------------------------------------------------

        for face in faces:

            embedding = getattr(
                face,
                "embedding",
                None
            )

            if embedding is None:
                continue

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            )

            # ------------------------------------------------
            # Normalize embedding
            # ------------------------------------------------

            norm = np.linalg.norm(
                embedding
            )

            if norm <= 0:
                continue

            embedding = (
                embedding / norm
            )

            bbox = getattr(
                face,
                "bbox",
                None
            )

            results.append(
                {
                    "bbox": bbox,
                    "embedding": embedding
                }
            )

        print(
            f"[FACE] Processed: "
            f"{image_path} | "
            f"Faces={len(results)}"
        )

        return results

    except Exception as e:

        print(
            f"[FACE ERROR] "
            f"{image_path}: "
            f"{repr(e)}"
        )

        return []


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_image = "test.jpg"

    print(
        "========================================"
    )

    print(
        "ClickMe Face Recognition Test"
    )

    print(
        "========================================"
    )

    faces = get_face_embeddings(
        test_image
    )

    print(
        f"Detected faces: {len(faces)}"
    )