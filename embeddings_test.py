from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import requests
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI()

# Load model once at startup (not per request)
print("Loading sentence-transformers model...", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully", flush=True)

# Warm up model with a dummy embedding to prevent first-request slowness
print("Warming up model...", flush=True)
_ = model.encode("warmup text", show_progress_bar=False)
print("Model warmed up and ready", flush=True)

# ✅ Secure configuration
QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY2NjcwMTM1fQ.c2bNP_BNXhVhM3fApCyKHw7SGV1ITyDMDtT5s1WlGW8"

# Collection names
SEGMENTS_COLLECTION = "video_transcript_segments"
LEGACY_COLLECTION = "text_embeddings"  # Keep old collection for backward compatibility

def get_qdrant_headers():
    headers = {"Content-Type": "application/json"}
    if QDRANT_API_KEY:
        headers["api-key"] = QDRANT_API_KEY
    return headers

def create_segments_collection():
    """Create the video_transcript_segments collection with proper schema"""
    try:
        check_url = f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}"
        response = requests.get(check_url, headers=get_qdrant_headers())
        
        if response.status_code == 404:
            print(f"Creating collection '{SEGMENTS_COLLECTION}'...", flush=True)
            create_url = f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}"
            data = {
                "vectors": {
                    "size": 384,  # all-MiniLM-L6-v2 dimension
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "indexing_threshold": 1000
                },
                # Define payload schema for better filtering
                "payload_schema": {
                    "video_id": {"type": "integer", "index": True},
                    "speaker": {"type": "keyword", "index": True},
                    "diarization_speaker": {"type": "keyword", "index": True},
                    "start_time": {"type": "float", "index": True},
                    "end_time": {"type": "float", "index": True},
                    "language": {"type": "keyword", "index": True}
                }
            }
            create_response = requests.put(create_url, json=data, headers=get_qdrant_headers())
            create_response.raise_for_status()
            print(f"Collection '{SEGMENTS_COLLECTION}' created successfully", flush=True)
        else:
            print(f"Collection '{SEGMENTS_COLLECTION}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)
        raise

def create_legacy_collection():
    """Create the legacy text_embeddings collection"""
    try:
        check_url = f"{QDRANT_URL}/collections/{LEGACY_COLLECTION}"
        response = requests.get(check_url, headers=get_qdrant_headers())
        
        if response.status_code == 404:
            print(f"Creating legacy collection '{LEGACY_COLLECTION}'...", flush=True)
            create_url = f"{QDRANT_URL}/collections/{LEGACY_COLLECTION}"
            data = {
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                },
                "optimizers_config": {
                    "indexing_threshold": 10000
                }
            }
            create_response = requests.put(create_url, json=data, headers=get_qdrant_headers())
            create_response.raise_for_status()
            print(f"Collection '{LEGACY_COLLECTION}' created successfully", flush=True)
    except Exception as e:
        print(f"Error managing legacy collection: {str(e)}", flush=True)

# Initialize collections
create_segments_collection()
create_legacy_collection()

@app.post("/embed-video")
async def embed_video(data: dict):
    """
    Embed an entire video's transcript as individual segments
    Expected data structure:
    {
        "video_id": int,
        "video_title": str,
        "video_filename": str,
        "youtube_url": str,
        "language": str,
        "identification_segments": [...],
        "speakers_transcript": [...],
        "diarization_segments": [...]
    }
    """
    try:
        video_id = data.get("video_id")
        if not video_id:
            raise HTTPException(status_code=400, detail="video_id is required")
        
        identification_segments = data.get("identification_segments", [])
        if not identification_segments:
            raise HTTPException(status_code=400, detail="identification_segments is required")
        
        video_title = data.get("video_title", "")
        video_filename = data.get("video_filename", "")
        youtube_url = data.get("youtube_url", "")
        language = data.get("language", "")
        
        print(f"Processing video {video_id} with {len(identification_segments)} segments", flush=True)
        
        # Debug: Log first segment structure
        if identification_segments:
            first_seg = identification_segments[0]
            print(f"DEBUG: First segment keys: {list(first_seg.keys())}", flush=True)
            print(f"DEBUG: First segment has 'text' key: {'text' in first_seg}", flush=True)
            if 'text' in first_seg:
                print(f"DEBUG: First segment text length: {len(first_seg.get('text', ''))}", flush=True)
                print(f"DEBUG: First segment text preview: {first_seg.get('text', '')[:100]}", flush=True)
            else:
                print(f"DEBUG: First segment full content: {first_seg}", flush=True)
        
        # First, delete existing embeddings for this video (if any)
        delete_existing_embeddings(video_id)
        
        # Prepare points for batch insertion
        points = []
        segments_embedded = 0
        segments_without_text = 0
        
        # Collect all texts first for batch embedding
        texts_to_embed = []
        segment_metadata = []
        
        for idx, segment in enumerate(identification_segments):
            # Extract segment data
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            diarization_speaker = segment.get("diarizationSpeaker", "")
            match_type = segment.get("match", "")
            confidence = segment.get("confidence", 0)
            
            # Skip empty segments
            if not text or len(text.strip()) < 3:
                segments_without_text += 1
                if idx < 3:  # Log first 3 skipped segments
                    print(f"DEBUG: Skipping segment {idx}: speaker={speaker}, start={start_time}, end={end_time}, text='{text[:50] if text else 'EMPTY'}'", flush=True)
                continue
            
            # Store text and metadata for batch processing
            texts_to_embed.append(text)
            segment_metadata.append({
                'idx': idx,
                'speaker': speaker,
                'diarization_speaker': diarization_speaker,
                'match_type': match_type,
                'start_time': start_time,
                'end_time': end_time,
                'confidence': confidence,
                'text': text
            })
        
        if not texts_to_embed:
            print(f"DEBUG: No valid segments found. Total segments: {len(identification_segments)}, Segments without text: {segments_without_text}", flush=True)
            raise HTTPException(status_code=400, detail=f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        # Generate embeddings for ALL texts in one batch (MUCH faster!)
        print(f"Generating embeddings for {len(texts_to_embed)} segments in batch...", flush=True)
        batch_start_time = datetime.utcnow()
        vectors = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32).tolist()
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        print(f"Batch embedding completed in {batch_duration:.2f} seconds", flush=True)
        
        # Create points with pre-computed embeddings
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            
            # Create unique ID for this segment (Qdrant accepts string or UUID)
            import hashlib
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
            
            # Prepare metadata payload
            payload = {
                "video_id": video_id,
                "video_title": video_title,
                "video_filename": video_filename,
                "youtube_url": youtube_url,
                "language": language,
                "segment_index": metadata['idx'],
                "speaker": metadata['speaker'],
                "diarization_speaker": metadata['diarization_speaker'],
                "match_type": metadata['match_type'],
                "start_time": metadata['start_time'],
                "end_time": metadata['end_time'],
                "duration": metadata['end_time'] - metadata['start_time'],
                "text": metadata['text'],
                "text_length": len(metadata['text']),
                "confidence": metadata['confidence'],
                "created_at": datetime.utcnow().isoformat()
            }
            
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": payload
            })
            
            segments_embedded += 1
        
        if not points:
            print(f"DEBUG: No valid segments found. Total segments: {len(identification_segments)}, Segments without text: {segments_without_text}", flush=True)
            raise HTTPException(status_code=400, detail=f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        # Batch insert all points (Qdrant handles batches efficiently)
        batch_size = 100
        total_inserted = 0
        
        print(f"Inserting {len(points)} points into Qdrant in batches of {batch_size}...", flush=True)
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            
            batch_payload = {"points": batch}
            
            try:
                print(f"Inserting batch {i//batch_size + 1}/{(len(points) + batch_size - 1)//batch_size}...", flush=True)
                response = requests.put(
                    f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}/points",
                    json=batch_payload,
                    headers=get_qdrant_headers(),
                    timeout=30
                )
                response.raise_for_status()
                total_inserted += len(batch)
                print(f"✓ Batch {i//batch_size + 1} inserted successfully ({total_inserted}/{len(points)} total)", flush=True)
            except requests.exceptions.HTTPError as e:
                print(f"ERROR: Qdrant rejected batch {i//batch_size + 1}", flush=True)
                print(f"Status code: {e.response.status_code}", flush=True)
                print(f"Response: {e.response.text}", flush=True)
                print(f"Sample point from batch: {batch[0] if batch else 'empty'}", flush=True)
                raise HTTPException(status_code=500, detail=f"Qdrant error: {e.response.text}")
        
        return {
            "success": True,
            "video_id": video_id,
            "collection": SEGMENTS_COLLECTION,
            "segments_embedded": segments_embedded,
            "total_points_inserted": total_inserted,
            "message": f"Successfully embedded {segments_embedded} segments for video {video_id}"
        }
        
    except Exception as e:
        print(f"Error in embed_video: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

def delete_existing_embeddings(video_id: int):
    """Delete all existing embeddings for a video"""
    try:
        delete_payload = {
            "filter": {
                "must": [
                    {"key": "video_id", "match": {"value": video_id}}
                ]
            }
        }
        
        response = requests.post(
            f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}/points/delete",
            json=delete_payload,
            headers=get_qdrant_headers(),
            timeout=10
        )
        
        if response.ok:
            print(f"Deleted existing embeddings for video {video_id}", flush=True)
        else:
            print(f"No existing embeddings to delete for video {video_id}", flush=True)
            
    except Exception as e:
        print(f"Error deleting embeddings: {str(e)}", flush=True)

@app.post("/embed")
async def embed(data: dict):
    """Legacy endpoint - kept for backward compatibility"""
    text = data.get("text", "")
    video_id = data.get("video_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Generate embedding
    vector = model.encode(text).tolist()
    vector_id = f"video_{video_id}_{uuid.uuid4()}" if video_id else str(uuid.uuid4())

    metadata = {
        "text": text[:5000],  # Truncate very long texts
        "text_length": len(text),
        "created_at": datetime.utcnow().isoformat(),
        "source": "legacy_embedding_api"
    }
    
    if video_id:
        metadata["video_id"] = video_id

    payload = {
        "points": [{
            "id": vector_id,
            "vector": vector,
            "payload": metadata
        }]
    }
    
    try:
        response = requests.put(
            f"{QDRANT_URL}/collections/{LEGACY_COLLECTION}/points",
            json=payload,
            headers=get_qdrant_headers(),
            timeout=10
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
      
    return {
        "id": vector_id,
        "embedding": vector,
        "vector_dimension": len(vector),
        "metadata": metadata,
        "status": "success"
    }

@app.post("/search")
async def search(data: dict):
    """Search for similar segments across all videos or within a specific video"""
    query_text = data.get("query", "")
    top_k = data.get("top_k", 10)
    video_id_filter = data.get("video_id")
    speaker_filter = data.get("speaker")
    min_score = data.get("min_score", 0.5)  # Minimum similarity score
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    query_vector = model.encode(query_text).tolist()
    
    search_payload = {
        "vector": query_vector,
        "limit": top_k,
        "with_payload": True,
        "with_vector": False,
        "score_threshold": min_score  # Only return results above this threshold
    }
    
    # Build filters
    filters = []
    if video_id_filter:
        filters.append({"key": "video_id", "match": {"value": video_id_filter}})
    if speaker_filter:
        filters.append({"key": "speaker", "match": {"value": speaker_filter}})
    
    if filters:
        search_payload["filter"] = {"must": filters}
    
    try:
        response = requests.post(
            f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}/points/search",
            json=search_payload,
            headers=get_qdrant_headers(),
            timeout=15
        )
        response.raise_for_status()
        results = response.json()
        
        return {
            "query": query_text,
            "collection": SEGMENTS_COLLECTION,
            "filters_applied": {
                "video_id": video_id_filter,
                "speaker": speaker_filter,
                "min_score": min_score
            },
            "results": [
                {
                    "id": r["id"],
                    "score": round(r["score"], 4),
                    "video_id": r["payload"].get("video_id"),
                    "video_title": r["payload"].get("video_title", ""),
                    "speaker": r["payload"].get("speaker", ""),
                    "diarization_speaker": r["payload"].get("diarization_speaker", ""),
                    "start_time": r["payload"].get("start_time", 0),
                    "end_time": r["payload"].get("end_time", 0),
                    "text": r["payload"].get("text", ""),
                    "youtube_url": r["payload"].get("youtube_url", ""),
                    "created_at": r["payload"].get("created_at")
                }
                for r in results.get("result", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}/segments")
async def get_video_segments(video_id: int, limit: int = 100):
    """Get all embedded segments for a specific video"""
    try:
        scroll_payload = {
            "filter": {
                "must": [
                    {"key": "video_id", "match": {"value": video_id}}
                ]
            },
            "limit": limit,
            "with_payload": True,
            "with_vector": False
        }
        
        response = requests.post(
            f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}/points/scroll",
            json=scroll_payload,
            headers=get_qdrant_headers(),
            timeout=10
        )
        response.raise_for_status()
        results = response.json()
        
        points = results.get("result", {}).get("points", [])
        
        return {
            "video_id": video_id,
            "total_segments": len(points),
            "segments": [
                {
                    "speaker": p["payload"].get("speaker"),
                    "start_time": p["payload"].get("start_time"),
                    "end_time": p["payload"].get("end_time"),
                    "text": p["payload"].get("text"),
                }
                for p in points
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/video/{video_id}/embeddings")
async def delete_video_embeddings(video_id: int):
    """Delete all embeddings for a specific video"""
    try:
        delete_existing_embeddings(video_id)
        return {
            "success": True,
            "message": f"Deleted all embeddings for video {video_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}",
            headers=get_qdrant_headers(),
            timeout=5
        )
        return {
            "status": "healthy",
            "qdrant_connected": response.ok,
            "collections": {
                "segments": SEGMENTS_COLLECTION,
                "legacy": LEGACY_COLLECTION
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/stats")
async def stats():
    """Get collection statistics"""
    try:
        response = requests.get(
            f"{QDRANT_URL}/collections/{SEGMENTS_COLLECTION}",
            headers=get_qdrant_headers(),
            timeout=5
        )
        response.raise_for_status()
        collection_info = response.json()
        
        return {
            "collection": SEGMENTS_COLLECTION,
            "points_count": collection_info.get("result", {}).get("points_count", 0),
            "vectors_count": collection_info.get("result", {}).get("vectors_count", 0),
            "indexed_vectors_count": collection_info.get("result", {}).get("indexed_vectors_count", 0),
            "status": collection_info.get("result", {}).get("status", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"🚀 Starting FastAPI Video Embedding Service on port {port}...", flush=True)
    print(f"📊 Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"📦 Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
