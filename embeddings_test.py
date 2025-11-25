from email.mime import text
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import requests
import uuid
import json
import os

app = FastAPI()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Qdrant endpoint - supports both local and cloud
QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY ="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY2NjcwMTM1fQ.c2bNP_BNXhVhM3fApCyKHw7SGV1ITyDMDtT5s1WlGW8"
COLLECTION_NAME = "text_embeddings"

# Helper function to get headers with API key for Qdrant Cloud
def get_qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers

# Create collection if it doesn't exist
def create_collection():
    """Create the text_embeddings collection if it doesn't exist"""
    try:
        # Check if collection exists
        check_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
        response = requests.get(check_url, headers=get_qdrant_headers())
        
        if response.status_code == 404:
            # Collection doesn't exist, create it
            print(f"Collection '{COLLECTION_NAME}' not found. Creating...", flush=True)
            create_url = f"{QDRANT_URL}/collections/{COLLECTION_NAME}"
            data = {
                "vectors": {
                    "size": 384,  # all-MiniLM-L6-v2 embeddings
                    "distance": "Cosine"
                }
            }
            create_response = requests.put(create_url, json=data, headers=get_qdrant_headers())
            create_response.raise_for_status()
            print(f"Collection '{COLLECTION_NAME}' created successfully", flush=True)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)

create_collection()

@app.post("/embed")
async def embed(data: dict):
    # log the incoming request
    print(f"Received embed request: {data}")
    text = data.get("text", "")
    video_id = data.get("video_id")  # Optional video ID for metadata

    print("text to embed:", text, flush=True)
    
    if not text:
        return {"error": "Text is required"}, 400
    
    # Generate embedding using sentence-transformers
    print("About to encode...", flush=True)
    vector = model.encode(text).tolist()
    print("Encoding complete",vector, flush=True)

    # Use a unique ID for this embedding
    vector_id = str(uuid.uuid4())

    # Prepare metadata
    metadata = {
        "text": text[:500],  # Store first 500 chars to avoid too large payload
        "text_length": len(text),
        "created_at": str(uuid.uuid1().time)
    }
    
    if video_id:
        metadata["video_id"] = video_id

    # Store embedding in Qdrant using the correct API format
    payload = {
        "points": [
            {
                "id": vector_id,
                "vector": vector,
                "payload": metadata
            }
        ]
    }
    
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
            json=payload,
            headers=get_qdrant_headers()
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # If collection not found, try to create it and retry
        if "not found" in str(e).lower() or "404" in str(e):
            print(f"Collection not found during embed. Attempting to create...", flush=True)
            create_collection()
            # Retry the request
            try:
                response = requests.put(
                    f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                    json=payload,
                    headers=get_qdrant_headers()
                )
                response.raise_for_status()
            except requests.exceptions.RequestException as retry_error:
                return {
                    "error": f"Failed to store in Qdrant after creating collection: {str(retry_error)}",
                    "vector_id": vector_id,
                    "embedding": vector
                }
        else:
            return {
                "error": f"Failed to store in Qdrant: {str(e)}",
                "vector_id": vector_id,
                "embedding": vector
            }
      
    return {
        "id": vector_id,
        "embedding": vector,
        "qdrant_response": response.json() if response.ok else response.text
    }



# Run the application when executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"Starting FastAPI server on port {port}...", flush=True)
    print(f"Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"Qdrant API Key configured: {bool(QDRANT_API_KEY)}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
