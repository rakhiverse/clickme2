import insightface
import cv2
import numpy as np

app = None


def get_app():
    global app

    if app is None:
        print("Loading InsightFace buffalo_s model...")

        app = insightface.app.FaceAnalysis(
            name="buffalo_s",
            allowed_modules=["detection", "recognition"]
        )

        # Render Free = CPU only
        # Smaller detection size reduces RAM/CPU usage
        app.prepare(
            ctx_id=-1,
            det_size=(320, 320)
        )

        print("InsightFace model loaded successfully.")

    return app


def get_face_embeddings(image_path):
    try:
        img = cv2.imread(image_path)

        if img is None:
            print(f"Image load nahi hui: {image_path}")
            return []

        h, w = img.shape[:2]

        # Very large photos ko resize karo
        if max(h, w) > 1600:
            scale = 1600.0 / max(h, w)

            new_w = int(w * scale)
            new_h = int(h * scale)

            img = cv2.resize(
                img,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

        face_app = get_app()

        faces = face_app.get(img)

        results = []

        for face in faces:
            emb = face.embedding

            # Normalize embedding
            norm = np.linalg.norm(emb)

            if norm > 0:
                emb = emb / norm

            results.append({
                "bbox": face.bbox,
                "embedding": emb
            })

        print(
            f"Processed {image_path}: "
            f"{len(results)} face(s) detected"
        )

        return results

    except Exception as e:
        print(f"Face processing error for {image_path}: {e}")
        return []


if __name__ == "__main__":
    test_image = "test.jpg"

    faces = get_face_embeddings(test_image)

    print(f"{len(faces)} face(s) detect hue")