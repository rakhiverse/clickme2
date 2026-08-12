import insightface
import cv2
import numpy as np

app = None

def get_app():
    global app
    if app is None:
        # Load ONLY detection & recognition modules to fit within 512MB RAM free tier
        app = insightface.app.FaceAnalysis(
            name='buffalo_s',
            allowed_modules=['detection', 'recognition']
        )
        app.prepare(ctx_id=-1, det_size=(640,640))
    return app

def get_face_embeddings(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image load nahi hui: {image_path}")
        return []
    h, w = img.shape[:2]
    if max(h, w) > 2048:
        scale = 2048.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    face_app = get_app()
    faces = face_app.get(img)
    results = []
    for face in faces:
        emb = face.embedding
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        results.append({
            "bbox": face.bbox,
            "embedding": emb
        })
    return results

if __name__ == "__main__":
    test_image = "test.jpg"
    faces = get_face_embeddings(test_image)
    print(f"{len(faces)} face(s) detect hue")