from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchParams
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI()

# Load model once at startup (not per request)
print("Loading sentence-transformers model.. .", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully", flush=True)

# Warm up model with a dummy embedding to prevent first-request slowness
print("Warming up model.. .", flush=True)
_ = model.encode("warmup text", show_progress_bar=False)
print("Model warmed up and ready", flush=True)

# ✅ Secure configuration
QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY2NjcwMTM1fQ.c2bNP_BNXhVhM3fApCyKHw7SGV1ITyDMDtT5s1WlGW8"

# Collection names
SEGMENTS_COLLECTION = "video_transcript_segments"
LEGACY_COLLECTION = "text_embeddings"

# Initialize Qdrant Client (MUCH better than requests!)
print("Connecting to Qdrant...", flush=True)
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)
print("✓ Connected to Qdrant successfully", flush=True)

def create_segments_collection():
    """Create the video_transcript_segments collection with proper schema"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections(). collections
        collection_names = [c.name for c in collections]
        
        if SEGMENTS_COLLECTION not in collection_names:
            print(f"Creating collection '{SEGMENTS_COLLECTION}'...", flush=True)
            
            qdrant_client.create_collection(
                collection_name=SEGMENTS_COLLECTION,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance. COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=1000
                )
            )
            
            # Create payload indexes for faster filtering
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="video_id",
                field_schema=models.PayloadSchemaType.INTEGER
            )
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="speaker",
                field_schema=models.PayloadSchemaType. KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=SEGMENTS_COLLECTION,
                field_name="start_time",
                field_schema=models. PayloadSchemaType.FLOAT
            )
            
            print(f"✓ Collection '{SEGMENTS_COLLECTION}' created successfully", flush=True)
        else:
            print(f"✓ Collection '{SEGMENTS_COLLECTION}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)
        raise

def create_legacy_collection():
    """Create the legacy text_embeddings collection"""
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if LEGACY_COLLECTION not in collection_names:
            print(f"Creating legacy collection '{LEGACY_COLLECTION}'...", flush=True)
            
            qdrant_client. create_collection(
                collection_name=LEGACY_COLLECTION,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance. COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            
            print(f"✓ Collection '{LEGACY_COLLECTION}' created successfully", flush=True)
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
        "identification_segments": [... ],
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
                continue
            
            # Store text and metadata for batch processing
            texts_to_embed. append(text)
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
            raise HTTPException(
                status_code=400, 
                detail=f"No valid segments found to embed.  Total: {len(identification_segments)}, Without text: {segments_without_text}"
            )
        
        # Generate embeddings for ALL texts in one batch (MUCH faster!)
        print(f"Generating embeddings for {len(texts_to_embed)} segments in batch...", flush=True)
        batch_start_time = datetime.utcnow()
        vectors = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32). tolist()
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        print(f"✓ Batch embedding completed in {batch_duration:. 2f} seconds", flush=True)
        
        # Create points with pre-computed embeddings
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            
            # Create unique ID for this segment
            import hashlib
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid. NAMESPACE_DNS, id_string))
            
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
            
            # Use PointStruct for Qdrant client
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )
            
            segments_embedded += 1
        
        if not points:
            raise HTTPException(
                status_code=400, 
                detail=f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}"
            )
        
        # Batch insert all points using Qdrant client (handles batching automatically!)
        print(f"Inserting {len(points)} points into Qdrant...", flush=True)
        
        try:
            # Qdrant client automatically handles batching efficiently
            qdrant_client. upsert(
                collection_name=SEGMENTS_COLLECTION,
                points=points,
                wait=True  # Wait for indexing to complete
            )
            print(f"✓ Successfully inserted {len(points)} points", flush=True)
        except Exception as e:
            print(f"ERROR: Qdrant insertion failed: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")
        
        return {
            "success": True,
            "video_id": video_id,
            "collection": SEGMENTS_COLLECTION,
            "segments_embedded": segments_embedded,
            "total_points_inserted": len(points),
            "embedding_time_seconds": round(batch_duration, 2),
            "message": f"Successfully embedded {segments_embedded} segments for video {video_id}"
        }
        
    except Exception as e:
        print(f"Error in embed_video: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

def delete_existing_embeddings(video_id: int):
    """Delete all existing embeddings for a video"""
    try:
        qdrant_client.delete(
            collection_name=SEGMENTS_COLLECTION,
            points_selector=models.FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="video_id",
                            match=MatchValue(value=video_id)
                        )
                    ]
                )
            )
        )
        print(f"✓ Deleted existing embeddings for video {video_id}", flush=True)
    except Exception as e:
        print(f"Note: Could not delete embeddings (may not exist): {str(e)}", flush=True)

@app. post("/embed")
async def embed(data: dict):
    """Legacy endpoint - kept for backward compatibility"""
    text = data.get("text", "")
    video_id = data.get("video_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Generate embedding
    vector = model.encode(text). tolist()
    vector_id = f"video_{video_id}_{uuid.uuid4()}" if video_id else str(uuid.uuid4())

    metadata = {
        "text": text[:5000],  # Truncate very long texts
        "text_length": len(text),
        "created_at": datetime.utcnow().isoformat(),
        "source": "legacy_embedding_api"
    }
    
    if video_id:
        metadata["video_id"] = video_id

    try:
        qdrant_client.upsert(
            collection_name=LEGACY_COLLECTION,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=vector,
                    payload=metadata
                )
            ]
        )
    except Exception as e:
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
    """
    🔍 SEMANTIC SEARCH - Find similar segments using AI-powered similarity
    
    Searches for segments semantically similar to the query text.
    
    Parameters:
    - query: Text to search for (required)
    - top_k: Number of results to return (default: 10)
    - video_id: Filter by specific video (optional)
    - speaker: Filter by speaker name (optional)
    - min_score: Minimum similarity score 0-1 (default: 0.5)
    - language: Filter by language (optional)
    - time_range: Filter by time range {"start": float, "end": float} (optional)
    """
    query_text = data.get("query", "")
    top_k = data.get("top_k", 10)
    video_id_filter = data.get("video_id")
    speaker_filter = data.get("speaker")
    language_filter = data.get("language")
    min_score = data.get("min_score", 0.5)
    time_range = data.get("time_range")  # {"start": 0, "end": 100}
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    print(f"🔍 Semantic search for: '{query_text[:100]}'", flush=True)
    
    # Generate query embedding
    query_vector = model.encode(query_text).tolist()
    
    # Build filters
    filter_conditions = []
    
    if video_id_filter:
        filter_conditions.append(
            FieldCondition(
                key="video_id",
                match=MatchValue(value=video_id_filter)
            )
        )
    
    if speaker_filter:
        filter_conditions.append(
            FieldCondition(
                key="speaker",
                match=MatchValue(value=speaker_filter)
            )
        )
    
    if language_filter:
        filter_conditions.append(
            FieldCondition(
                key="language",
                match=MatchValue(value=language_filter)
            )
        )
    
    # Time range filter
    if time_range:
        start_time = time_range.get("start")
        end_time = time_range.get("end")
        
        if start_time is not None:
            filter_conditions.append(
                FieldCondition(
                    key="start_time",
                    range=models.Range(gte=start_time)
                )
            )
        
        if end_time is not None:
            filter_conditions.append(
                FieldCondition(
                    key="end_time",
                    range=models. Range(lte=end_time)
                )
            )
    
    # Build filter object
    search_filter = None
    if filter_conditions:
        search_filter = Filter(must=filter_conditions)
    
    try:
        # Perform semantic search using Qdrant client
        search_results = qdrant_client.search(
            collection_name=SEGMENTS_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=min_score,  # Only return results above this threshold
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False  # Don't return vectors (saves bandwidth)
        )
        
        print(f"✓ Found {len(search_results)} results", flush=True)
        
        return {
            "query": query_text,
            "collection": SEGMENTS_COLLECTION,
            "total_results": len(search_results),
            "filters_applied": {
                "video_id": video_id_filter,
                "speaker": speaker_filter,
                "language": language_filter,
                "time_range": time_range,
                "min_score": min_score
            },
            "results": [
                {
                    "id": r.id,
                    "score": round(r.score, 4),
                    "similarity_percentage": round(r.score * 100, 2),
                    "video_id": r.payload.get("video_id"),
                    "video_title": r.payload.get("video_title", ""),
                    "speaker": r.payload.get("speaker", ""),
                    "diarization_speaker": r.payload.get("diarization_speaker", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "duration": round(r.payload.get("end_time", 0) - r.payload.get("start_time", 0), 2),
                    "text": r.payload.get("text", ""),
                    "text_length": r.payload.get("text_length", 0),
                    "youtube_url": r.payload.get("youtube_url", ""),
                    "language": r.payload.get("language", ""),
                    "created_at": r.payload.get("created_at"),
                    # Create timestamped YouTube URL
                    "youtube_url_timestamped": f"{r.payload.get('youtube_url', '')}?t={int(r.payload.get('start_time', 0))}" if r.payload.get('youtube_url') else ""
                }
                for r in search_results
            ]
        }
    except Exception as e:
        print(f"ERROR in search: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-multi-video")
async def search_multi_video(data: dict):
    """
    🔍 Search across multiple specific videos
    
    Parameters:
    - query: Text to search for (required)
    - video_ids: List of video IDs to search in (required)
    - top_k: Number of results per video (default: 5)
    - min_score: Minimum similarity score (default: 0.5)
    """
    query_text = data. get("query", "")
    video_ids = data.get("video_ids", [])
    top_k = data.get("top_k", 5)
    min_score = data.get("min_score", 0. 5)
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    if not video_ids:
        raise HTTPException(status_code=400, detail="video_ids list is required")
    
    print(f"🔍 Searching across {len(video_ids)} videos for: '{query_text[:100]}'", flush=True)
    
    query_vector = model.encode(query_text).tolist()
    
    try:
        # Search with video_id filter using "should" (OR condition)
        search_filter = Filter(
            should=[
                FieldCondition(
                    key="video_id",
                    match=MatchValue(value=vid)
                )
                for vid in video_ids
            ]
        )
        
        search_results = qdrant_client.search(
            collection_name=SEGMENTS_COLLECTION,
            query_vector=query_vector,
            limit=top_k * len(video_ids),  # Get more results to cover all videos
            score_threshold=min_score,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        )
        
        # Group results by video_id
        results_by_video = {}
        for r in search_results:
            vid = r.payload.get("video_id")
            if vid not in results_by_video:
                results_by_video[vid] = []
            
            if len(results_by_video[vid]) < top_k:  # Limit per video
                results_by_video[vid].append({
                    "id": r. id,
                    "score": round(r.score, 4),
                    "similarity_percentage": round(r.score * 100, 2),
                    "speaker": r.payload.get("speaker", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "text": r.payload.get("text", ""),
                    "youtube_url_timestamped": f"{r. payload.get('youtube_url', '')}?t={int(r.payload.get('start_time', 0))}" if r. payload.get('youtube_url') else ""
                })
        
        return {
            "query": query_text,
            "total_videos_searched": len(video_ids),
            "videos_with_results": len(results_by_video),
            "results_by_video": results_by_video
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/video/{video_id}/segments")
async def get_video_segments(video_id: int, limit: int = 100, offset: int = 0):
    """Get all embedded segments for a specific video with pagination"""
    try:
        # Use scroll for efficient retrieval
        scroll_result = qdrant_client.scroll(
            collection_name=SEGMENTS_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="video_id",
                        match=MatchValue(value=video_id)
                    )
                ]
            ),
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = scroll_result
        
        return {
            "video_id": video_id,
            "total_segments": len(points),
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset,
            "segments": [
                {
                    "segment_index": p.payload.get("segment_index"),
                    "speaker": p.payload.get("speaker"),
                    "start_time": p.payload.get("start_time"),
                    "end_time": p.payload.get("end_time"),
                    "duration": p.payload.get("duration"),
                    "text": p.payload.get("text"),
                    "text_length": p.payload.get("text_length"),
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
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": {
                "segments": SEGMENTS_COLLECTION,
                "legacy": LEGACY_COLLECTION
            },
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "qdrant_connected": False,
            "error": str(e)
        }

@app.get("/stats")
async def stats():
    """Get collection statistics"""
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        
        return {
            "collection": SEGMENTS_COLLECTION,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info. vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "status": collection_info. status,
            "optimizer_status": collection_info.optimizer_status,
            "config": {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params. vectors.distance. name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """API documentation"""
    return {
        "service": "Video Transcript Semantic Search API",
        "version": "2.0",
        "endpoints": {
            "POST /embed-video": "Embed entire video transcript",
            "POST /search": "Semantic search across all videos or filtered by video_id/speaker",
            "POST /search-multi-video": "Search across multiple specific videos",
            "GET /video/{video_id}/segments": "Get all segments for a video",
            "DELETE /video/{video_id}/embeddings": "Delete all embeddings for a video",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics"
        },
        "features": [
            "✅ Semantic search using sentence transformers",
            "✅ Batch embedding for performance",
            "✅ Qdrant client for efficient operations",
            "✅ Advanced filtering (video, speaker, time range, language)",
            "✅ Score thresholding",
            "✅ Timestamped YouTube URLs"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"🚀 Starting FastAPI Video Embedding Service on port {port}...", flush=True)
    print(f"📊 Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"📦 Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
