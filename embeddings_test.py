from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from qdrant_client import models, QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchParams
import uuid
import os
from datetime import datetime
from typing import Optional, List, Dict

app = FastAPI()

print("Loading sentence-transformers model.. .", flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded successfully", flush=True)

print("Warming up model...", flush=True)
_ = model.encode("warmup text", show_progress_bar=False)
print("Model warmed up and ready", flush=True)

QDRANT_URL = "https://558d3fea-5962-46da-bffa-94aba210a6c6.eu-west-1-0.aws.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzY2NjcwMTM1fQ.c2bNP_BNXhVhM3fApCyKHw7SGV1ITyDMDtT5s1WlGW8"

SEGMENTS_COLLECTION = "video_transcript_segments"
LEGACY_COLLECTION = "text_embeddings"

print("Connecting to Qdrant.. .", flush=True)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
print("Connected to Qdrant successfully", flush=True)

def create_segments_collection():
    try:
        collections = qdrant_client.get_collections(). collections
        collection_names = [c.name for c in collections]
        
        if SEGMENTS_COLLECTION not in collection_names:
            print(f"Creating collection '{SEGMENTS_COLLECTION}'...", flush=True)
            qdrant_client.create_collection(
                collection_name=SEGMENTS_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000)
            )
            qdrant_client.create_payload_index(collection_name=SEGMENTS_COLLECTION, field_name="video_id", field_schema=models.PayloadSchemaType.INTEGER)
            qdrant_client.create_payload_index(collection_name=SEGMENTS_COLLECTION, field_name="speaker", field_schema=models.PayloadSchemaType. KEYWORD)
            qdrant_client.create_payload_index(collection_name=SEGMENTS_COLLECTION, field_name="start_time", field_schema=models.PayloadSchemaType. FLOAT)
            print(f"Collection '{SEGMENTS_COLLECTION}' created successfully", flush=True)
        else:
            print(f"Collection '{SEGMENTS_COLLECTION}' already exists", flush=True)
    except Exception as e:
        print(f"Error managing collection: {str(e)}", flush=True)
        raise

def create_legacy_collection():
    try:
        collections = qdrant_client. get_collections().collections
        collection_names = [c.name for c in collections]
        
        if LEGACY_COLLECTION not in collection_names:
            print(f"Creating legacy collection '{LEGACY_COLLECTION}'...", flush=True)
            qdrant_client.create_collection(
                collection_name=LEGACY_COLLECTION,
                vectors_config=VectorParams(size=384, distance=Distance. COSINE),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10000)
            )
            print(f"Collection '{LEGACY_COLLECTION}' created successfully", flush=True)
    except Exception as e:
        print(f"Error managing legacy collection: {str(e)}", flush=True)

create_segments_collection()
create_legacy_collection()

@app.post("/embed-video")
async def embed_video(data: dict):
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
        
        delete_existing_embeddings(video_id)
        
        points = []
        segments_embedded = 0
        segments_without_text = 0
        texts_to_embed = []
        segment_metadata = []
        
        for idx, segment in enumerate(identification_segments):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "")
            start_time = float(segment.get("start", 0))
            end_time = float(segment.get("end", 0))
            diarization_speaker = segment.get("diarizationSpeaker", "")
            match_type = segment.get("match", "")
            confidence = segment.get("confidence", 0)
            
            if not text or len(text. strip()) < 3:
                segments_without_text += 1
                continue
            
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
            raise HTTPException(status_code=400, detail=f"No valid segments found to embed.  Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        print(f"Generating embeddings for {len(texts_to_embed)} segments in batch.. .", flush=True)
        batch_start_time = datetime.utcnow()
        vectors = model.encode(texts_to_embed, show_progress_bar=False, batch_size=32). tolist()
        batch_end_time = datetime.utcnow()
        batch_duration = (batch_end_time - batch_start_time).total_seconds()
        print(f"Batch embedding completed in {batch_duration:.2f} seconds", flush=True)
        
        for i, metadata in enumerate(segment_metadata):
            vector = vectors[i]
            id_string = f"video_{video_id}_seg_{metadata['idx']}"
            point_id = str(uuid.uuid5(uuid. NAMESPACE_DNS, id_string))
            
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
            
            points.append(PointStruct(id=point_id, vector=vector, payload=payload))
            segments_embedded += 1
        
        if not points:
            raise HTTPException(status_code=400, detail=f"No valid segments found to embed. Total: {len(identification_segments)}, Without text: {segments_without_text}")
        
        print(f"Inserting {len(points)} points into Qdrant...", flush=True)
        
        try:
            qdrant_client.upsert(collection_name=SEGMENTS_COLLECTION, points=points, wait=True)
            print(f"Successfully inserted {len(points)} points", flush=True)
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
    try:
        qdrant_client.delete(
            collection_name=SEGMENTS_COLLECTION,
            points_selector=models.FilterSelector(
                filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))])
            )
        )
        print(f"Deleted existing embeddings for video {video_id}", flush=True)
    except Exception as e:
        print(f"Note: Could not delete embeddings (may not exist): {str(e)}", flush=True)

@app.post("/embed")
async def embed(data: dict):
    text = data.get("text", "")
    video_id = data.get("video_id")
    
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    vector = model.encode(text). tolist()
    vector_id = f"video_{video_id}_{uuid.uuid4()}" if video_id else str(uuid.uuid4())

    metadata = {
        "text": text[:5000],
        "text_length": len(text),
        "created_at": datetime.utcnow().isoformat(),
        "source": "legacy_embedding_api"
    }
    
    if video_id:
        metadata["video_id"] = video_id

    try:
        qdrant_client.upsert(collection_name=LEGACY_COLLECTION, points=[PointStruct(id=vector_id, vector=vector, payload=metadata)])
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
    Combined semantic + optional keyword search.

    Accepts:
      - query (str): semantic query (required unless `word` provided)
      - word (str): exact/substring word to search for in payload.text (optional)
      - top_k (int): number of results to return (default 10)
      - video_id, speaker, language: filters (optional)
      - min_score (float): semantic score threshold (default 0.5)
      - time_range (dict): {"start": float, "end": float} (optional)
      - max_scanned (int): limit for number of items scanned for keyword search (default 10000)
    """
    query_text = data.get("query", "")
    word = data.get("word")  # optional substring/word search
    top_k = int(data.get("top_k", 10))
    video_id_filter = data.get("video_id")
    speaker_filter = data.get("speaker")
    language_filter = data.get("language")
    min_score = float(data.get("min_score", 0.5))
    time_range = data.get("time_range")
    max_scanned = int(data.get("max_scanned", 10000))  # avoids unbounded scans

    if not query_text and not word:
        raise HTTPException(status_code=400, detail="Either 'query' or 'word' is required")

    # Build Qdrant payload filter conditions
    filter_conditions = []
    if video_id_filter is not None:
        filter_conditions.append(FieldCondition(key="video_id", match=MatchValue(value=video_id_filter)))
    if speaker_filter is not None:
        filter_conditions.append(FieldCondition(key="speaker", match=MatchValue(value=speaker_filter)))
    if language_filter is not None:
        filter_conditions.append(FieldCondition(key="language", match=MatchValue(value=language_filter)))
    if time_range:
        start_time = time_range.get("start")
        end_time = time_range.get("end")
        if start_time is not None:
            filter_conditions.append(FieldCondition(key="start_time", range=models.Range(gte=start_time)))
        if end_time is not None:
            filter_conditions.append(FieldCondition(key="end_time", range=models.Range(lte=end_time)))

    search_filter = Filter(must=filter_conditions) if filter_conditions else None

    semantic_results = []
    keyword_results = []

    # Run semantic search if query_text present
    if query_text:
        try:
            print(f"Semantic search for: '{query_text[:120]}'", flush=True)
            query_vector = model.encode(query_text).tolist()

            sem_search_results = qdrant_client.search(
                collection_name=SEGMENTS_COLLECTION,
                query_vector=query_vector,
                limit=top_k * 3,  # fetch a few more to allow later dedupe/boost
                score_threshold=min_score,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False
            )

            for r in sem_search_results:
                semantic_results.append({
                    "id": r.id,
                    "score": float(getattr(r, "score", 0.0)),
                    "video_id": r.payload.get("video_id"),
                    "video_title": r.payload.get("video_title", ""),
                    "speaker": r.payload.get("speaker", ""),
                    "diarization_speaker": r.payload.get("diarization_speaker", ""),
                    "start_time": r.payload.get("start_time", 0),
                    "end_time": r.payload.get("end_time", 0),
                    "duration": round((r.payload.get("end_time", 0) - r.payload.get("start_time", 0)), 2),
                    "text": r.payload.get("text", ""),
                    "text_length": r.payload.get("text_length", 0),
                    "youtube_url": r.payload.get("youtube_url", ""),
                    "language": r.payload.get("language", ""),
                    "created_at": r.payload.get("created_at"),
                    "match_types": ["semantic"]
                })

        except Exception as e:
            print(f"ERROR during semantic search: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

    # Run keyword/word (substring) search if requested
    if word:
        try:
            print(f"Keyword search for word: '{word}' (max_scanned={max_scanned})", flush=True)
            word_lower = word.lower()
            page_size = 1000
            scanned = 0
            offset = 0

            # If we have a filter that narrows to video_id only, use it to speed up scroll.
            scroll_filter = None
            if video_id_filter is not None:
                scroll_filter = Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id_filter))])
            elif filter_conditions:
                # if there are other filters, reuse combined filter to limit scan
                scroll_filter = search_filter

            while scanned < max_scanned:
                # fetch a page of points
                points, next_offset = qdrant_client.scroll(
                    collection_name=SEGMENTS_COLLECTION,
                    scroll_filter=scroll_filter,
                    limit=min(page_size, max_scanned - scanned),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not points:
                    break

                for p in points:
                    scanned += 1
                    text = (p.payload.get("text") or "")
                    if word_lower in text.lower():
                        keyword_results.append({
                            "id": p.id,
                            "score": 1.0,  # keyword exactness gets top score
                            "video_id": p.payload.get("video_id"),
                            "video_title": p.payload.get("video_title", ""),
                            "speaker": p.payload.get("speaker", ""),
                            "diarization_speaker": p.payload.get("diarization_speaker", ""),
                            "start_time": p.payload.get("start_time", 0),
                            "end_time": p.payload.get("end_time", 0),
                            "duration": round((p.payload.get("end_time", 0) - p.payload.get("start_time", 0)), 2),
                            "text": text,
                            "text_length": p.payload.get("text_length", 0),
                            "youtube_url": p.payload.get("youtube_url", ""),
                            "language": p.payload.get("language", ""),
                            "created_at": p.payload.get("created_at"),
                            "match_types": ["keyword"]
                        })
                        # stop early if we have enough keyword matches
                        if len(keyword_results) >= top_k:
                            break

                if len(keyword_results) >= top_k:
                    break

                # If API returned next_offset, use it; otherwise increment offset
                if not next_offset:
                    # If no next_offset provided, we increment offset by page_size
                    offset += page_size
                else:
                    offset = next_offset

                # Stop if we've scanned up to max_scanned
                if scanned >= max_scanned:
                    break

        except Exception as e:
            print(f"ERROR during keyword search: {str(e)}", flush=True)
            raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")

    # Merge semantic and keyword results (deduplicate by id). Keyword matches get priority.
    merged = {}
    for r in semantic_results:
        merged[r["id"]] = r

    for r in keyword_results:
        if r["id"] in merged:
            # if already present as semantic, add keyword to match_types and boost score
            if "keyword" not in merged[r["id"]]["match_types"]:
                merged[r["id"]]["match_types"].append("keyword")
            # boost score so exact keyword matches rank higher
            merged[r["id"]]["score"] = max(merged[r["id"]]["score"], r["score"])
        else:
            merged[r["id"]] = r

    # Convert to list and sort by score descending, then limit to top_k
    merged_list = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Build response
    return {
        "query": query_text,
        "word": word,
        "collection": SEGMENTS_COLLECTION,
        "total_semantic_hits": len(semantic_results),
        "total_keyword_hits": len(keyword_results),
        "returned": len(merged_list),
        "filters_applied": {
            "video_id": video_id_filter,
            "speaker": speaker_filter,
            "language": language_filter,
            "time_range": time_range,
            "min_score": min_score,
            "max_scanned_for_keyword": max_scanned
        },
        "results": [
            {
                "id": r["id"],
                "score": round(r["score"], 4),
                "match_types": r.get("match_types", []),
                "video_id": r.get("video_id"),
                "video_title": r.get("video_title", ""),
                "speaker": r.get("speaker", ""),
                "diarization_speaker": r.get("diarization_speaker", ""),
                "start_time": r.get("start_time", 0),
                "end_time": r.get("end_time", 0),
                "duration": r.get("duration", 0),
                "text": r.get("text", ""),
                "text_length": r.get("text_length", 0),
                "youtube_url": r.get("youtube_url", ""),
                "language": r.get("language", ""),
                "created_at": r.get("created_at"),
                "youtube_url_timestamped": f"{r.get('youtube_url', '')}?t={int(r.get('start_time', 0))}" if r.get('youtube_url') else ""
            }
            for r in merged_list
        ]
    }
@app.post("/search-multi-video")
async def search_multi_video(data: dict):
    query_text = data.get("query", "")
    video_ids = data.get("video_ids", [])
    top_k = data.get("top_k", 5)
    min_score = data.get("min_score", 0.5)
    
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is required")
    
    if not video_ids:
        raise HTTPException(status_code=400, detail="video_ids list is required")
    
    print(f"Searching across {len(video_ids)} videos for: '{query_text[:100]}'", flush=True)
    
    query_vector = model. encode(query_text).tolist()
    
    try:
        search_filter = Filter(should=[FieldCondition(key="video_id", match=MatchValue(value=vid)) for vid in video_ids])
        
        search_results = qdrant_client.search(
            collection_name=SEGMENTS_COLLECTION,
            query_vector=query_vector,
            limit=top_k * len(video_ids),
            score_threshold=min_score,
            query_filter=search_filter,
            with_payload=True,
            with_vectors=False
        )
        
        results_by_video = {}
        for r in search_results:
            vid = r.payload.get("video_id")
            if vid not in results_by_video:
                results_by_video[vid] = []
            
            if len(results_by_video[vid]) < top_k:
                results_by_video[vid]. append({
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
    try:
        scroll_result = qdrant_client.scroll(
            collection_name=SEGMENTS_COLLECTION,
            scroll_filter=Filter(must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]),
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
    try:
        delete_existing_embeddings(video_id)
        return {"success": True, "message": f"Deleted all embeddings for video {video_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        return {
            "status": "healthy",
            "qdrant_connected": True,
            "collections": {"segments": SEGMENTS_COLLECTION, "legacy": LEGACY_COLLECTION},
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count
        }
    except Exception as e:
        return {"status": "unhealthy", "qdrant_connected": False, "error": str(e)}

@app.get("/stats")
async def stats():
    try:
        collection_info = qdrant_client.get_collection(collection_name=SEGMENTS_COLLECTION)
        return {
            "collection": SEGMENTS_COLLECTION,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info. vectors_count,
            "indexed_vectors_count": collection_info.indexed_vectors_count,
            "status": collection_info.status,
            "optimizer_status": collection_info.optimizer_status,
            "config": {
                "vector_size": collection_info.config.params.vectors. size,
                "distance": collection_info.config.params.vectors.distance. name
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
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
            "Semantic search using sentence transformers",
            "Batch embedding for performance",
            "Qdrant client for efficient operations",
            "Advanced filtering (video, speaker, time range, language)",
            "Score thresholding",
            "Timestamped YouTube URLs"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 9000))
    print(f"Starting FastAPI Video Embedding Service on port {port}...", flush=True)
    print(f"Qdrant URL: {QDRANT_URL}", flush=True)
    print(f"Collections: {SEGMENTS_COLLECTION}, {LEGACY_COLLECTION}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port)
