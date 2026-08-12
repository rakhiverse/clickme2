import os
import chromadb
from face_utils import get_face_embeddings

# ChromaDB client - local folder mein data save hoga
client = chromadb.PersistentClient(path="./clickme_db")
collection = client.get_or_create_collection(name="event_faces")

def process_event_folder(folder_path, event_id):
    count = 0
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder_path, filename)
        faces = get_face_embeddings(path)
        for i, face in enumerate(faces):
            doc_id = f"{event_id}_{filename}_{i}"
            collection.add(
                ids=[doc_id],
                embeddings=[face["embedding"].tolist()],
                metadatas=[{"event_id": event_id, "filename": filename}]
            )
            count += 1
        print(f"{filename}: {len(faces)} face(s) processed")
    print(f"\nTotal {count} face embeddings stored for event '{event_id}'")

if __name__ == "__main__":
    process_event_folder("test_event", "event001")