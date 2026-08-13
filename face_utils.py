import os

# ============================================================
# RENDER FREE / LOW MEMORY SETTINGS
# IMPORTANT: set these BEFORE importing numpy / cv2 / insightface
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ONNX Runtime CPU thread control
os.environ["ORT_INTRA_OP_NUM_THREADS"] = "1"
os.environ["ORT_INTER_OP_NUM_THREADS"] = "1"

# Disable unnecessary ONNX logging
os.environ["ORT_LOGGING_LEVEL"] = "3"

# No GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc

import cv2
import numpy as np
import insightface


# ============================================================
# GLOBAL FACE APP
# ============================================================

app = None


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def get_app():
    global app

    if app is not None:
        return app

    print("[FACE] Loading InsightFace buffalo_s model...")

    try:
        # Only the two modules actually needed by ClickMe
        app = insightface.app.FaceAnalysis(
            name="buffalo_s",
            allowed_modules=[
                "detection",
                "recognition",
            ],
        )

        # CPU only
        #
        # 256x256 instead of 320x320:
        # lower RAM / CPU usage on Render Free.
        app.prepare(
            ctx_id=-1,
            det_size=(256, 256),
        )

        print("[FACE] InsightFace buffalo_s loaded successfully.")

        return app

    except Exception as e:
        print(
            "[FACE ERROR] Could not load InsightFace: "
            f"{repr(e)}"
        )

        app = None

        # Release anything partially allocated
        gc.collect()

        raise


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path: str):
    try:
        img = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR,
        )

        if img is None:
            print(
                "[FACE] Image could not be loaded: "
                f"{image_path}"
            )
            return None

        return img

    except Exception as e:
        print(
            "[FACE] Image loading error: "
            f"{repr(e)}"
        )
        return None


# ============================================================
# RESIZE IMAGE
# ============================================================

def resize_image(img):
    """
    Resize large images to reduce RAM and CPU usage.

    Maximum dimension:
        1000 px
    """

    try:
        height, width = img.shape[:2]

        max_dimension = max(height, width)

        if max_dimension <= 1000:
            return img

        scale = 1000.0 / float(max_dimension)

        new_width = max(
            1,
            int(width * scale),
        )

        new_height = max(
            1,
            int(height * scale),
        )

        resized = cv2.resize(
            img,
            (
                new_width,
                new_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        print(
            "[FACE] Image resized: "
            f"{width}x{height} -> "
            f"{new_width}x{new_height}"
        )

        return resized

    except Exception as e:
        print(
            "[FACE] Resize error: "
            f"{repr(e)}"
        )

        return img


# ============================================================
# GET FACE EMBEDDINGS
# ============================================================

def get_face_embeddings(image_path: str):
    img = None

    try:
        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        img = load_image(image_path)

        if img is None:
            return []

        # ----------------------------------------------------
        # Resize large image
        # ----------------------------------------------------

        img = resize_image(img)

        # ----------------------------------------------------
        # Load face model
        # ----------------------------------------------------

        face_app = get_app()

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        faces = face_app.get(img)

        if not faces:
            print(
                "[FACE] No face detected: "
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
                None,
            )

            if embedding is None:
                continue

            # Convert to float32
            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            # Normalize embedding
            norm = np.linalg.norm(embedding)

            if norm <= 0:
                continue

            embedding = embedding / norm

            bbox = getattr(
                face,
                "bbox",
                None,
            )

            # Convert bbox to normal Python list.
            # This avoids keeping unnecessary InsightFace
            # objects alive.
            if bbox is not None:
                bbox = np.asarray(
                    bbox,
                    dtype=np.float32,
                ).tolist()

            results.append(
                {
                    "bbox": bbox,
                    "embedding": embedding,
                }
            )

        print(
            "[FACE] Processed: "
            f"{image_path} | "
            f"Faces={len(results)}"
        )

        return results

    except Exception as e:
        print(
            "[FACE ERROR] "
            f"{image_path}: "
            f"{repr(e)}"
        )

        return []

    finally:
        # Release image memory after every photo.
        img = None
        gc.collect()


# ============================================================
# SIMPLE TEST
# ============================================================

if __name__ == "__main__":

    test_image = "test.jpg"

    print("========================================")
    print("ClickMe Face Recognition Test")
    print("========================================")

    try:
        faces = get_face_embeddings(test_image)

        print(
            f"Detected faces: {len(faces)}"
        )

    except Exception as e:
        print(
            "[TEST ERROR]",
            repr(e),
        )