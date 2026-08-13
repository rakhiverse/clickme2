import os
import shutil
import uuid
from typing import List

# ============================================================
# LOW MEMORY SETTINGS
# ============================================================

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import chromadb

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    BackgroundTasks
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from face_utils import get_face_embeddings


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ClickMe API"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

DB_DIR = os.path.join(
    BASE_DIR,
    "clickme_db"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    DB_DIR,
    exist_ok=True
)


# ============================================================
# STATIC UPLOADS
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(
        directory=UPLOAD_DIR
    ),
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
    allow_headers=["*"]
)


# ============================================================
# CHROMADB
# ============================================================

_client = None
_collection = None


def get_collection():

    global _client
    global _collection

    if _collection is None:

        print(
            "[CHROMA] Initializing ChromaDB..."
        )

        _client = chromadb.PersistentClient(
            path=DB_DIR
        )

        _collection = (
            _client.get_or_create_collection(
                name="event_faces"
            )
        )

        print(
            "[CHROMA] ChromaDB ready."
        )

    return _collection


# ============================================================
# SAFE FILENAME
# ============================================================

def safe_filename(filename: str):

    if not filename:
        return ""

    return os.path.basename(
        filename
    )


# ============================================================
# BACKGROUND FACE PROCESSING
# ============================================================

def process_photo_faces(
    save_path: str,
    event_id: str,
    filename: str
):

    try:

        print(
            f"[BACKGROUND] Processing: "
            f"{filename} | "
            f"event={event_id}"
        )

        faces = get_face_embeddings(
            save_path
        )

        if not faces:

            print(
                f"[BACKGROUND] "
                f"No face detected: "
                f"{filename}"
            )

            return

        collection = get_collection()

        for index, face in enumerate(
            faces
        ):

            embedding = face[
                "embedding"
            ]

            doc_id = (
                f"{event_id}_"
                f"{uuid.uuid4().hex}_"
                f"{index}"
            )

            collection.add(
                ids=[doc_id],

                embeddings=[
                    embedding.tolist()
                ],

                metadatas=[
                    {
                        "event_id": event_id,
                        "filename": filename
                    }
                ]
            )

        print(
            f"[BACKGROUND] Successfully "
            f"indexed {len(faces)} face(s): "
            f"{filename}"
        )

    except Exception as e:

        print(
            f"[BACKGROUND ERROR] "
            f"{filename}: {repr(e)}"
        )


# ============================================================
# UPLOAD EVENT PHOTOS
# ============================================================

@app.post(
    "/upload-event-photos"
)
async def upload_event_photos(

    background_tasks: BackgroundTasks,

    event_id: str = Form(...),

    files: List[
        UploadFile
    ] = File(...)
):

    event_id = event_id.strip()

    if not event_id:

        return {
            "status": "error",
            "message": "Event ID is required."
        }

    uploaded_count = 0
    scheduled_count = 0

    for file in files:

        if not file.filename:
            continue

        original_filename = (
            safe_filename(
                file.filename
            )
        )

        if not original_filename:
            continue

        save_path = os.path.join(
            UPLOAD_DIR,
            original_filename
        )

        try:

            print(
                f"[UPLOAD] Saving: "
                f"{original_filename}"
            )

            with open(
                save_path,
                "wb"
            ) as output_file:

                shutil.copyfileobj(
                    file.file,
                    output_file
                )

            uploaded_count += 1

            # Schedule processing after
            # HTTP response work.
            background_tasks.add_task(
                process_photo_faces,
                save_path,
                event_id,
                original_filename
            )

            scheduled_count += 1

            print(
                f"[UPLOAD] Background task "
                f"scheduled: "
                f"{original_filename}"
            )

        except Exception as e:

            print(
                f"[UPLOAD ERROR] "
                f"{original_filename}: "
                f"{repr(e)}"
            )

    return {
        "status": "success",
        "files_count": uploaded_count,
        "faces_processed": 0,
        "tasks_scheduled": scheduled_count,
        "message": (
            "Roll received. "
            "Photos are being processed "
            "in background."
        )
    }


# ============================================================
# FIND MY PHOTOS
# ============================================================

@app.post(
    "/find-my-photos"
)
async def find_my_photos(

    event_id: str = Form(...),

    selfie: UploadFile = File(...),

    threshold: float = Form(
        0.85
    )
):

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

    selfie_filename = safe_filename(
        selfie.filename
    )

    selfie_name = (
        f"selfie_"
        f"{uuid.uuid4().hex}_"
        f"{selfie_filename}"
    )

    selfie_path = os.path.join(
        UPLOAD_DIR,
        selfie_name
    )

    try:

        # ----------------------------------------------------
        # Save selfie
        # ----------------------------------------------------

        with open(
            selfie_path,
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                selfie.file,
                output_file
            )

        print(
            f"[SELFIE] Processing: "
            f"{selfie_name}"
        )

        # ----------------------------------------------------
        # Detect guest face
        # ----------------------------------------------------

        guest_faces = get_face_embeddings(
            selfie_path
        )

        if not guest_faces:

            return {
                "status": "error",
                "message": (
                    "No face detected in "
                    "the selfie. "
                    "Please try another photo."
                )
            }

        guest_embedding = (
            guest_faces[0][
                "embedding"
            ]
        ).tolist()

        # ----------------------------------------------------
        # Validate threshold
        # ----------------------------------------------------

        try:

            threshold = float(
                threshold
            )

        except Exception:

            threshold = 0.85

        if threshold <= 0:
            threshold = 0.85

        if threshold > 2:
            threshold = 0.85

        # ----------------------------------------------------
        # ChromaDB
        # ----------------------------------------------------

        collection = get_collection()

        total = collection.count()

        print(
            f"[SEARCH] Total embeddings: "
            f"{total}"
        )

        if total == 0:

            return {
                "status": "success",
                "matched_photos": [],
                "message": (
                    "No processed event "
                    "photos found yet."
                )
            }

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        n_results = min(
            50,
            total
        )

        results = collection.query(

            query_embeddings=[
                guest_embedding
            ],

            n_results=n_results,

            where={
                "event_id": event_id
            }
        )

        matched = set()

        metadatas = []

        distances = []

        if (
            results
            and results.get(
                "metadatas"
            )
        ):

            metadatas = (
                results[
                    "metadatas"
                ][0]
            )

        if (
            results
            and results.get(
                "distances"
            )
        ):

            distances = (
                results[
                    "distances"
                ][0]
            )

        # ----------------------------------------------------
        # Match faces
        # ----------------------------------------------------

        for metadata, distance in zip(
            metadatas,
            distances
        ):

            if not metadata:
                continue

            if distance is None:
                continue

            print(
                f"[SEARCH] "
                f"{metadata.get('filename')} "
                f"distance={distance:.4f}"
            )

            if distance < threshold:

                filename = metadata.get(
                    "filename"
                )

                if filename:
                    matched.add(
                        filename
                    )

        matched_photos = list(
            matched
        )

        print(
            f"[SEARCH] "
            f"Event={event_id} | "
            f"Matches={len(matched_photos)}"
        )

        return {
            "status": "success",
            "matched_photos": (
                matched_photos
            )
        }

    except Exception as e:

        print(
            f"[SEARCH ERROR] "
            f"{repr(e)}"
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

@app.get(
    "/",
    response_class=HTMLResponse
)
def root():

    frontend_path = os.path.join(
        BASE_DIR,
        "clickme_frontend.html"
    )

    try:

        with open(
            frontend_path,
            "r",
            encoding="utf-8"
        ) as file:

            return HTMLResponse(
                content=file.read()
            )

    except Exception as e:

        return HTMLResponse(
            content=(
                "<h1>ClickMe</h1>"
                "<p>Frontend file could not "
                f"be loaded: {e}</p>"
            ),
            status_code=500
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health"
)
def health():

    return {
        "status": "ok"
    }


# ============================================================
# DEBUG STATUS
# ============================================================

@app.get(
    "/debug/status"
)
def debug_status():

    try:

        collection = get_collection()

        count = collection.count()

        return {
            "status": "ok",
            "database": "connected",
            "embeddings": count
        }

    except Exception as e:

        return {
            "status": "error",
            "database": "error",
            "message": repr(e)
        }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )