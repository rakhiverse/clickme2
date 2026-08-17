import os
import re
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
)

from fastapi.responses import HTMLResponse, FileResponse
from face_utils import get_face_embeddings


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ClickMe API",
    version="1.0.0"
)


# ============================================================
# SECURITY LIMITS
# ============================================================

MAX_EVENT_PHOTOS = 2000
MAX_PHOTO_SIZE = 10 * 1024 * 1024      # 10 MB
MAX_SELFIE_SIZE = 5 * 1024 * 1024      # 5 MB

# Server-controlled face matching threshold.
# Clients must not be allowed to weaken matching security.
MATCH_THRESHOLD = 0.95

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

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
# CONTROLLED PHOTO ACCESS
# ============================================================

@app.get("/photo/{event_id}/{filename}")
def get_photo(event_id: str, filename: str):

    if not is_valid_event_id(event_id):
        return {
            "status": "error",
            "message": "Invalid Event ID."
        }

    filename = safe_filename(filename)

    if not filename:
        return {
            "status": "error",
            "message": "Invalid filename."
        }

    try:
        collection = get_collection()

        results = collection.get(
            where={
                "$and": [
                    {"event_id": event_id},
                    {"filename": filename}
                ]
            },
            include=["metadatas"]
        )

        metadatas = results.get("metadatas") or []

        stored_filename = None

        for metadata in metadatas:
            if metadata:
                stored_filename = metadata.get(
                    "stored_filename"
                )
                if stored_filename:
                    break

        if not stored_filename:
            return {
                "status": "error",
                "message": "Photo not found."
            }

        stored_filename = safe_filename(
            stored_filename
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            stored_filename
        )

        if not os.path.isfile(file_path):
            return {
                "status": "error",
                "message": "Photo not found."
            }

        return FileResponse(file_path)

    except Exception as e:

        print(
            f"[PHOTO ERROR] "
            f"event={event_id} "
            f"filename={filename}: "
            f"{repr(e)}"
        )

        return {
            "status": "error",
            "message": "Photo could not be loaded."
        }

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
# SECURITY VALIDATION
# ============================================================

def is_valid_event_id(event_id: str):
    if not event_id:
        return False

    return re.fullmatch(
        r"[A-Za-z0-9_-]{1,100}",
        event_id
    ) is not None


def is_allowed_image(filename: str):

    if not filename:
        return False

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in ALLOWED_IMAGE_EXTENSIONS


def is_safe_upload_size(file_size: int, max_size: int):

    if file_size is None:
        return False

    return 0 < file_size <= max_size


def is_valid_image_content(file_path: str):

    try:

        import cv2

        image = cv2.imread(
            file_path,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return False

        height, width = image.shape[:2]

        return (
            height > 0
            and width > 0
        )

    except Exception as e:

        print(
            f"[SECURITY] Image validation error: "
            f"{repr(e)}"
        )

        return False


# ============================================================
# PROCESS ONE PHOTO
# ============================================================

def process_photo_faces(
    save_path: str,
    event_id: str,
    filename: str,
    stored_filename: str
):

    """
    Detect faces and immediately store embeddings
    in ChromaDB.

    Returns:
        faces_count, success, error_message
    """

    try:

        print(
            f"[FACE] Processing: "
            f"{filename} | event={event_id}"
        )

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        faces = get_face_embeddings(
            save_path
        )

        face_count = len(
            faces or []
        )

        print(
            f"[FACE] Processed: "
            f"{filename} | "
            f"Faces={face_count}"
        )

        if not faces:

            return 0, True, None

        # ----------------------------------------------------
        # CHROMA
        # ----------------------------------------------------

        collection = get_collection()

        added = 0

        for index, face in enumerate(
            faces
        ):

            embedding = face.get(
                "embedding"
            )

            if embedding is None:
                continue

            # Convert numpy array safely
            if hasattr(
                embedding,
                "tolist"
            ):
                embedding = embedding.tolist()

            # Stable ID:
            # same event + same filename + same face
            # will update the existing record instead
            # of creating a duplicate.
            doc_id = (
                f"{event_id}_"
                f"{filename}_"
                f"{index}"
            )

            collection.upsert(
                ids=[
                    doc_id
                ],

                embeddings=[
                    embedding
                ],

                metadatas=[
                    {
                        "event_id": event_id,
                        "filename": filename,
                        "stored_filename": stored_filename
                    }
                ]
            )

            added += 1

        print(
            f"[CHROMA] Indexed/updated "
            f"{added} face(s): "
            f"{filename}"
        )

        return added, True, None

    except Exception as e:

        print(
            f"[FACE ERROR] "
            f"{filename}: "
            f"{repr(e)}"
        )

        return 0, False, repr(e)


# ============================================================
# UPLOAD EVENT PHOTOS
# ============================================================

@app.post(
    "/upload-event-photos"
)
async def upload_event_photos(

    event_id: str = Form(...),

    files: List[
        UploadFile
    ] = File(...)
):

    event_id = event_id.strip()

    if not is_valid_event_id(event_id):

        return {
            "status": "error",
            "message": (
                "Invalid Event ID. Use only letters, "
                "numbers, hyphens, and underscores."
            )
        }

    if not files:

        return {
            "status": "error",
            "message": "No photos received."
        }

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    files_uploaded = 0

    faces_processed = 0

    files_without_faces = 0

    failed_files = 0

    failed_details = []

    # --------------------------------------------------------
    # SECURITY: LIMIT NUMBER OF EVENT PHOTOS
    # --------------------------------------------------------

    if len(files) > MAX_EVENT_PHOTOS:

        return {
            "status": "error",
            "message": (
                f"Too many photos. Maximum "
                f"{MAX_EVENT_PHOTOS} photos are allowed "
                "per upload."
            )
        }

    # --------------------------------------------------------
    # PROCESS EACH PHOTO
    # --------------------------------------------------------

    for file in files:

        if not file.filename:
            continue

        original_filename = safe_filename(
            file.filename
        )

        if not original_filename:
            continue

        # ----------------------------------------------------
        # SECURITY: IMAGE EXTENSION
        # ----------------------------------------------------

        if not is_allowed_image(
            original_filename
        ):

            failed_files += 1

            failed_details.append(
                {
                    "filename": original_filename,
                    "error": "Unsupported image file type."
                }
            )

            print(
                f"[SECURITY] Rejected file type: "
                f"{original_filename}"
            )

            continue

        # ----------------------------------------------------
        # SECURITY: FILE SIZE
        # ----------------------------------------------------

        try:

            file.file.seek(0, os.SEEK_END)

            file_size = file.file.tell()

            file.file.seek(0)

        except Exception as e:

            failed_files += 1

            failed_details.append(
                {
                    "filename": original_filename,
                    "error": "Could not determine file size."
                }
            )

            print(
                f"[SECURITY] Could not inspect size: "
                f"{original_filename}: {repr(e)}"
            )

            continue

        if not is_safe_upload_size(
            file_size,
            MAX_PHOTO_SIZE
        ):

            failed_files += 1

            failed_details.append(
                {
                    "filename": original_filename,
                    "error": "File is empty or exceeds the 10 MB limit."
                }
            )

            print(
                f"[SECURITY] Rejected file size: "
                f"{original_filename} "
                f"size={file_size} bytes"
            )

            continue

        # SECURITY: Use a unique physical filename so
        # repeated uploads cannot overwrite an existing file.
        stored_filename = (
            f"{uuid.uuid4().hex}_"
            f"{original_filename}"
        )

        save_path = os.path.join(
            UPLOAD_DIR,
            stored_filename
        )

        try:

            print(
                f"[UPLOAD] Saving: "
                f"{original_filename}"
            )

            # ------------------------------------------------
            # SAVE FILE
            # ------------------------------------------------

            with open(
                save_path,
                "wb"
            ) as output_file:

                shutil.copyfileobj(
                    file.file,
                    output_file
                )

            # ------------------------------------------------
            # SECURITY: VERIFY ACTUAL IMAGE CONTENT
            # ------------------------------------------------

            if not is_valid_image_content(
                save_path
            ):

                failed_files += 1

                failed_details.append(
                    {
                        "filename": original_filename,
                        "error": "File is not a valid image."
                    }
                )

                print(
                    f"[SECURITY] Rejected invalid image: "
                    f"{original_filename}"
                )

                try:
                    os.remove(save_path)
                except Exception:
                    pass

                continue

            files_uploaded += 1

            # ------------------------------------------------
            # PROCESS FACE IMMEDIATELY
            #
            # IMPORTANT:
            # Do NOT use BackgroundTasks here.
            #
            # We need the result before returning
            # the HTTP response.
            # ------------------------------------------------

            face_count, success, error = (
                process_photo_faces(
                    save_path,
                    event_id,
                    original_filename,
                    stored_filename
                )
            )

            if not success:

                failed_files += 1

                failed_details.append(
                    {
                        "filename": original_filename,
                        "error": error
                    }
                )

                continue

            if face_count == 0:

                files_without_faces += 1

            else:

                faces_processed += face_count

        except Exception as e:

            failed_files += 1

            error_text = repr(e)

            failed_details.append(
                {
                    "filename": original_filename,
                    "error": error_text
                }
            )

            print(
                f"[UPLOAD ERROR] "
                f"{original_filename}: "
                f"{error_text}"
            )

    # --------------------------------------------------------
    # FINAL DATABASE COUNT
    # --------------------------------------------------------

    try:

        collection = get_collection()

        total_embeddings = (
            collection.count()
        )

    except Exception:

        total_embeddings = 0

    print(
        "================================================"
    )

    print(
        "[ROLL COMPLETE]"
    )

    print(
        f"Event: {event_id}"
    )

    print(
        f"Files uploaded: {files_uploaded}"
    )

    print(
        f"Faces indexed: {faces_processed}"
    )

    print(
        f"Files without faces: "
        f"{files_without_faces}"
    )

    print(
        f"Failed files: {failed_files}"
    )

    print(
        f"Total embeddings: "
        f"{total_embeddings}"
    )

    print(
        "================================================"
    )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {
        "status": "success",

        "event_id": event_id,

        "files_uploaded": files_uploaded,

        "faces_processed": faces_processed,

        "files_without_faces": (
            files_without_faces
        ),

        "failed_files": failed_files,

        "failed_details": failed_details,

        "total_embeddings": (
            total_embeddings
        ),

        "message": (
            "Roll developed successfully. "
            "Faces are indexed and ready "
            "for search."
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

):

    event_id = event_id.strip()

    if not is_valid_event_id(event_id):

        return {
            "status": "error",
            "message": (
                "Invalid Event ID. Use only letters, "
                "numbers, hyphens, and underscores."
            )
        }

    if not selfie.filename:

        return {
            "status": "error",
            "message": "Selfie file is required."
        }

    selfie_filename = safe_filename(
        selfie.filename
    )

    if not selfie_filename:

        return {
            "status": "error",
            "message": "Invalid selfie filename."
        }

    # --------------------------------------------------------
    # SECURITY: SELFIE FILE TYPE
    # --------------------------------------------------------

    if not is_allowed_image(
        selfie_filename
    ):

        return {
            "status": "error",
            "message": (
                "Unsupported selfie file type. "
                "Please upload JPG, JPEG, PNG, or WEBP."
            )
        }

    # --------------------------------------------------------
    # SECURITY: SELFIE FILE SIZE
    # --------------------------------------------------------

    try:

        selfie.file.seek(0, os.SEEK_END)

        selfie_size = selfie.file.tell()

        selfie.file.seek(0)

    except Exception as e:

        print(
            f"[SECURITY] Could not inspect selfie size: "
            f"{repr(e)}"
        )

        return {
            "status": "error",
            "message": "Could not validate selfie file."
        }

    if not is_safe_upload_size(
        selfie_size,
        MAX_SELFIE_SIZE
    ):

        print(
            f"[SECURITY] Rejected selfie size: "
            f"{selfie_size} bytes"
        )

        return {
            "status": "error",
            "message": (
                "Selfie is empty or exceeds "
                "the 5 MB limit."
            )
        }

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
        # SAVE SELFIE
        # ----------------------------------------------------

        with open(
            selfie_path,
            "wb"
        ) as output_file:

            shutil.copyfileobj(
                selfie.file,
                output_file
            )

        # ----------------------------------------------------
        # SECURITY: VERIFY ACTUAL SELFIE IMAGE CONTENT
        # ----------------------------------------------------

        if not is_valid_image_content(
            selfie_path
        ):

            print(
                f"[SECURITY] Rejected invalid selfie: "
                f"{selfie_filename}"
            )

            try:
                os.remove(selfie_path)
            except Exception as e:
                print(
                    f"[SECURITY] Invalid selfie cleanup failed: "
                    f"{repr(e)}"
                )

            return {
                "status": "error",
                "message": (
                    "Uploaded selfie is not a valid image."
                )
            }

        print(
            f"[SELFIE] Processing: "
            f"{selfie_name}"
        )

        # ----------------------------------------------------
        # DETECT GUEST FACE
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
            guest_faces[0]
            .get("embedding")
        )

        if guest_embedding is None:

            return {
                "status": "error",
                "message": (
                    "Could not generate "
                    "face embedding."
                )
            }

        if hasattr(
            guest_embedding,
            "tolist"
        ):
            guest_embedding = (
                guest_embedding.tolist()
            )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------
        # SECURITY: Threshold is controlled by the server.
        # The client cannot weaken or alter matching rules.

        threshold = MATCH_THRESHOLD

        print(
            f"[SEARCH] Event={event_id} "
            f"Threshold={threshold}"
        )

        # ----------------------------------------------------
        # CHROMADB
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
                    "No event photos "
                    "have been indexed yet."
                )
            }

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        # Chroma requires n_results to be
        # no larger than available records.

        n_results = min(
            50,
            total
        )

        try:

            results = collection.query(

                query_embeddings=[
                    guest_embedding
                ],

                n_results=n_results,

                where={
                    "event_id": event_id
                },

                include=[
                    "metadatas",
                    "distances"
                ]
            )

        except Exception as e:

            print(
                f"[SEARCH QUERY ERROR] "
                f"{repr(e)}"
            )

            return {
                "status": "error",
                "message": (
                    "Face database search "
                    "failed."
                )
            }

        # ----------------------------------------------------
        # READ RESULTS
        # ----------------------------------------------------

        matched = set()

        metadatas = []

        distances = []

        if (
            results
            and results.get("metadatas")
        ):

            if results["metadatas"][0]:

                metadatas = (
                    results["metadatas"][0]
                )

        if (
            results
            and results.get("distances")
        ):

            if results["distances"][0]:

                distances = (
                    results["distances"][0]
                )

        # ----------------------------------------------------
        # MATCH FACES
        # ----------------------------------------------------

        for metadata, distance in zip(
            metadatas,
            distances
        ):

            if not metadata:
                continue

            if distance is None:
                continue

            filename = metadata.get(
                "filename"
            )

            print(
                f"[SEARCH] "
                f"{filename} "
                f"distance={distance:.4f}"
            )

            if distance <= threshold:

                if filename:

                    matched.add(
                        filename
                    )

        matched_photos = sorted(
            list(matched)
        )

        print(
            f"[SEARCH] "
            f"Event={event_id} | "
            f"Matches="
            f"{len(matched_photos)}"
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {
            "status": "success",

            "event_id": event_id,

            "matched_photos": (
                matched_photos
            ),

            "count": len(
                matched_photos
            ),

            "threshold": threshold
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

    finally:

        # ----------------------------------------------------
        # DELETE SELFIE AFTER PROCESSING
        # ----------------------------------------------------

        try:

            if os.path.exists(
                selfie_path
            ):

                os.remove(
                    selfie_path
                )

        except Exception as e:

            print(
                f"[SELFIE CLEANUP ERROR] "
                f"{repr(e)}"
            )


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
                "be loaded: "
                f"{e}</p>"
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

    try:

        collection = get_collection()

        return {
            "status": "ok",
            "database": "connected",
            "embeddings": collection.count()
        }

    except Exception as e:

        return {
            "status": "error",
            "database": "error",
            "message": repr(e)
        }


# ============================================================
# DEBUG STATUS
# ============================================================

DEBUG_STATUS_ENABLED = (
    os.environ.get(
        "DEBUG_STATUS_ENABLED",
        "false"
    ).lower()
    == "true"
)

if DEBUG_STATUS_ENABLED:

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
