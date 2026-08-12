from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import shutil
import os
import uuid
import chromadb

from face_utils import get_face_embeddings


# Disable Chroma telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"


app = FastAPI(title="ClickMe API")


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"
DB_DIR = "clickme_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)


# Static uploaded photos
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CHROMADB
# ============================================================

_client = None
_collection = None


def get_collection():
    global _client, _collection

    if _collection is None:
        print("Initializing ChromaDB...")

        _client = chromadb.PersistentClient(
            path=f"./{DB_DIR}"
        )

        _collection = _client.get_or_create_collection(
            name="event_faces"
        )

        print("ChromaDB ready.")

    return _collection


# ============================================================
# BACKGROUND FACE PROCESSING
# ============================================================

def process_photo_faces(
    save_path: str,
    event_id: str,
    filename: str
):
    """
    Runs AFTER upload response is sent.
    Face detection + embedding happens in background.
    """

    try:
        print(
            f"[BACKGROUND] Processing: {filename} "
            f"for event: {event_id}"
        )

        faces = get_face_embeddings(save_path)

        if not faces:
            print(
                f"[BACKGROUND] No face detected: {filename}"
            )
            return

        col = get_collection()

        for i, face in enumerate(faces):

            # Unique ID prevents Chroma duplicate ID errors
            doc_id = (
                f"{event_id}_{uuid.uuid4().hex}_{i}"
            )

            embedding = face["embedding"]

            col.add(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                metadatas=[
                    {
                        "event_id": event_id,
                        "filename": filename
                    }
                ]
            )

        print(
            f"[BACKGROUND] Successfully processed "
            f"{len(faces)} face(s): {filename}"
        )

    except Exception as e:

        print(
            f"[BACKGROUND ERROR] "
            f"{filename}: {repr(e)}"
        )


# ============================================================
# UPLOAD EVENT PHOTOS
# ============================================================

@app.post("/upload-event-photos")
async def upload_event_photos(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """
    Upload photographer photos.

    IMPORTANT:
    Face processing is scheduled in BackgroundTasks,
    so the HTTP request returns immediately.
    """

    event_id = event_id.strip()

    if not event_id:
        return {
            "status": "error",
            "message": "Event ID is required."
        }

    uploaded_count = 0

    for file in files:

        if not file.filename:
            continue

        # Keep filename safe
        original_filename = os.path.basename(
            file.filename
        )

        save_path = os.path.join(
            UPLOAD_DIR,
            original_filename
        )

        print(
            f"[UPLOAD] Saving {original_filename}"
        )

        try:

            with open(save_path, "wb") as f:
                shutil.copyfileobj(
                    file.file,
                    f
                )

            uploaded_count += 1

            # =================================================
            # IMPORTANT:
            # DO NOT call process_photo_faces() directly.
            # =================================================

            background_tasks.add_task(
                process_photo_faces,
                save_path,
                event_id,
                original_filename
            )

            print(
                f"[UPLOAD] Background task scheduled: "
                f"{original_filename}"
            )

        except Exception as e:

            print(
                f"[UPLOAD ERROR] "
                f"{original_filename}: {repr(e)}"
            )

    return {
        "status": "success",
        "faces_processed": 0,
        "files_count": uploaded_count,
        "message": (
            "Roll received! "
            "Photos are being processed in background."
        )
    }


# ============================================================
# FIND MY PHOTOS
# ============================================================

@app.post("/find-my-photos")
async def find_my_photos(
    event_id: str = Form(...),
    selfie: UploadFile = File(...),
    threshold: float = Form(0.85)
):
    """
    Guest uploads selfie and searches event photos.
    """

    event_id = event_id.strip()

    if not event_id:
        return {
            "status": "error",
            "message": "Event ID is required."
        }

    if not selfie.filename:
        return {
            "status": "error",
            "message": "Selfie file is required."
        }

    # Unique selfie filename
    selfie_name = (
        f"selfie_{uuid.uuid4().hex}_"
        f"{os.path.basename(selfie.filename)}"
    )

    selfie_path = os.path.join(
        UPLOAD_DIR,
        selfie_name
    )

    try:

        with open(selfie_path, "wb") as f:
            shutil.copyfileobj(
                selfie.file,
                f
            )

        print(
            f"[SELFIE] Processing: {selfie_name}"
        )

        guest_faces = get_face_embeddings(
            selfie_path
        )

        if not guest_faces:

            return {
                "status": "error",
                "message": (
                    "No face detected in the selfie. "
                    "Please try another photo."
                )
            }

        guest_embedding = (
            guest_faces[0]["embedding"].tolist()
        )

        # Safety for invalid threshold
        try:
            threshold = float(threshold)
        except Exception:
            threshold = 0.85

        if threshold <= 0:
            threshold = 0.85

        if threshold > 10:
            threshold = 0.85

        col = get_collection()

        # Check whether collection has data
        total = col.count()

        if total == 0:

            return {
                "status": "success",
                "matched_photos": [],
                "message": (
                    "No processed event photos found yet."
                )
            }

        # Don't request more results than available
        n_results = min(50, total)

        results = col.query(
            query_embeddings=[guest_embedding],
            n_results=n_results,
            where={
                "event_id": event_id
            }
        )

        matched = set()

        if (
            results
            and "metadatas" in results
            and results["metadatas"]
        ):

            metadatas = results["metadatas"][0]

            distances = []

            if (
                "distances" in results
                and results["distances"]
            ):
                distances = results["distances"][0]

            for metadata, distance in zip(
                metadatas,
                distances
            ):

                if (
                    metadata
                    and distance is not None
                    and distance < threshold
                ):
                    filename = metadata.get(
                        "filename"
                    )

                    if filename:
                        matched.add(filename)

        print(
            f"[SEARCH] Event={event_id}, "
            f"Matches={len(matched)}"
        )

        return {
            "status": "success",
            "matched_photos": list(matched)
        }

    except Exception as e:

        print(
            f"[SEARCH ERROR] {repr(e)}"
        )

        return {
            "status": "error",
            "message": (
                "Photo search failed. "
                "Please try again."
            )
        }


# ============================================================
# HOME
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():

    frontend_path = os.path.join(
        os.path.dirname(__file__),
        "clickme_frontend.html"
    )

    with open(
        frontend_path,
        "r",
        encoding="utf-8"
    ) as f:

        return HTMLResponse(
            content=f.read()
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )