import chromadb
from face_utils import get_face_embeddings

client = chromadb.PersistentClient(path="./clickme_db")
collection = client.get_or_create_collection(name="event_faces")

def find_matching_photos(selfie_path, event_id, threshold=0.5):
    guest_faces = get_face_embeddings(selfie_path)
    if not guest_faces:
        print("Selfie mein koi face detect nahi hua!")
        return []

    guest_embedding = guest_faces[0]["embedding"].tolist()

    results = collection.query(
        query_embeddings=[guest_embedding],
        n_results=28,
        where={"event_id": event_id}
    )

    matched = set()
    print("\n--- Sab results (distance ke saath) ---")
    for metadata, distance in zip(results["metadatas"][0], results["distances"][0]):
        print(f"{metadata['filename']}  ->  distance = {distance:.4f}")
        if distance < threshold:
            matched.add(metadata["filename"])

    print(f"\n--- MATCHED photos (threshold={threshold}) ---")
    for f in matched:
        print(f)
    return list(matched)

if __name__ == "__main__":
    find_matching_photos("my2.jpg", "event001", threshold=0.5)