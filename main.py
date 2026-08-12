from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

import shutil
import os
import chromadb

from face_utils import get_face_embeddings


os.environ["ANONYMIZED_TELEMETRY"] = "False"


app = FastAPI(title="ClickMe API")


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("clickme_db", exist_ok=True)


app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_DIR),
    name="uploads"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_client = None
_collection = None


def get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(
            path="./clickme_db"
        )

        _collection = _client.get_or_create_collection(
            name="event_faces"
        )

    return _collection


def process_photo_faces(
    save_path: str,
    event_id: str,
    filename: str
):
    """
    Background task:
    Extract face embeddings and save them to ChromaDB.
    """

    try:
        print(f"=== BACKGROUND PROCESSING START: {filename} ===")

        faces = get_face_embeddings(save_path)

        if faces:
            col = get_collection()

            for i, face in enumerate(faces):

                doc_id = f"{event_id}_{filename}_{i}"

                col.add(
                    ids=[doc_id],
                    embeddings=[
                        face["embedding"].tolist()
                    ],
                    metadatas=[
                        {
                            "event_id": event_id,
                            "filename": filename
                        }
                    ]
                )

            print(
                f"Successfully processed "
                f"{len(faces)} face(s) for {filename}"
            )

        else:
            print(
                f"No faces detected in {filename}"
            )

    except Exception as e:
        print(
            f"Error processing background face task "
            f"for {filename}: {e}"
        )


@app.post("/upload-event-photos")
async def upload_event_photos(
    background_tasks: BackgroundTasks,
    event_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    """
    Upload photos quickly.
    Face processing happens in background.
    """

    print("=== UPLOAD START ===")

    event_id = event_id.strip()

    uploaded_count = 0

    for file in files:

        print(
            f"=== RECEIVING FILE: {file.filename} ==="
        )

        save_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        with open(save_path, "wb") as f:
            shutil.copyfileobj(
                file.file,
                f
            )

        # IMPORTANT:
        # Do NOT process faces directly here.
        # Schedule it as a background task.
        background_tasks.add_task(
            process_photo_faces,
            save_path,
            event_id,
            file.filename
        )

        uploaded_count += 1

    print(
        f"=== UPLOAD COMPLETE: "
        f"{uploaded_count} files ==="
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


@app.post("/find-my-photos")
async def find_my_photos(
    event_id: str = Form(...),
    selfie: UploadFile = File(...),
    threshold: float = Form(0.85)
):
    """
    Guest selfie search endpoint.
    """

    event_id = event_id.strip()

    selfie_path = os.path.join(
        UPLOAD_DIR,
        f"selfie_{selfie.filename}"
    )

    with open(selfie_path, "wb") as f:
        shutil.copyfileobj(
            selfie.file,
            f
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

    effective_threshold = (
        0.85
        if threshold > 10.0
        else threshold
    )

    col = get_collection()

    results = col.query(
        query_embeddings=[guest_embedding],
        n_results=50,
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

        for metadata, distance in zip(
            results["metadatas"][0],
            results["distances"][0]
        ):

            if distance < effective_threshold:
                matched.add(
                    metadata["filename"]
                )

    return {
        "status": "success",
        "matched_photos": list(matched)
    }


@app.get(
    "/",
    response_class=HTMLResponse
)
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