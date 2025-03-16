"""
LanceDB Storage for Meeting Minutes
This module provides a storage implementation using LanceDB for vector search capabilities.
"""

import os
import json
import uuid
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
import lancedb
import numpy as np
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class LanceDBStorage:
    """
    Storage implementation using LanceDB for vector storage and search
    """
    
    def __init__(self, data_dir: str = "data", embedding_dim: int = 1536):
        """
        Initialize the LanceDB storage
        
        Args:
            data_dir: Directory to store the LanceDB database
            embedding_dim: Dimension of the embedding vectors (1536 for OpenAI embeddings)
        """
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim
        self.db_path = os.path.join(data_dir, "lancedb")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(self.db_path)
        
        # Initialize tables
        self.transcripts_table = self._get_or_create_table("transcripts")
        self.minutes_table = self._get_or_create_table("minutes")
    
    def _get_or_create_table(self, table_name: str):
        """
        Get or create a table in LanceDB
        
        Args:
            table_name: Name of the table
            
        Returns:
            LanceDB table
        """
        try:
            if table_name in self.db.table_names():
                return self.db.open_table(table_name)
            else:
                # Create a schema with a placeholder embedding for initialization
                # Create empty embedding vector for initialization
                data = [{
                    "id": "placeholder",
                    "content": "",
                    "metadata": json.dumps({}),
                    "embedding": np.zeros(self.embedding_dim).tolist()
                }]
                
                # Let LanceDB infer the schema from the data
                table = self.db.create_table(table_name, data=data)
                
                # Remove the placeholder after initialization
                table.delete("id = 'placeholder'")
                return table
        except Exception as e:
            logger.error(f"Error creating/opening table {table_name}: {str(e)}")
            raise
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Embedding as list of floats
        """
        try:
            # Try to use AnythingLLM for embeddings first
            try:
                from src.anything_llm.anything_llm_client import AnythingLLMClient
                
                # Get settings from environment or config
                anything_llm_url = os.environ.get("ANYTHING_LLM_URL") or self.config.get("anything_llm_url")
                anything_llm_api_key = os.environ.get("ANYTHING_LLM_API_KEY") or self.config.get("anything_llm_api_key")
                
                if anything_llm_url and anything_llm_api_key:
                    # Use the AnythingLLMClient class
                    client = AnythingLLMClient(
                        base_url=anything_llm_url,
                        api_key=anything_llm_api_key
                    )
                    
                    # Get embedding using the client
                    embedding = client.get_embedding(text)
                    if embedding:
                        return embedding
                    else:
                        logger.warning("AnythingLLM embedding failed, falling back to OpenAI")
            except Exception as e:
                logger.warning(f"Error using AnythingLLM for embeddings: {str(e)}")
            
            # Get OpenAI API key from environment or config
            api_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai_api_key")
            
            # If we have an API key, use OpenAI
            if api_key:
                try:
                    from openai import OpenAI
                    # Initialize the client without the proxies parameter
                    client = OpenAI(api_key=api_key)
                    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
                    return response.data[0].embedding
                except Exception as openai_error:
                    logger.error(f"OpenAI embedding error: {str(openai_error)}")
                    # Fall through to random embeddings
            else:
                logger.warning("No embedding providers available, using random embeddings")
                return np.random.rand(self.embedding_dim).tolist()
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Return random embedding as fallback
            return np.random.rand(self.embedding_dim).tolist()
    
    def store_transcript(self, transcript_text: str, metadata: Dict[str, Any], meeting_name: Optional[str] = None, 
                        duration: Optional[float] = None, source: str = "whisper") -> Dict[str, Any]:
        """
        Store a transcript in LanceDB
        
        Args:
            transcript_text: Transcript text
            metadata: Metadata for the transcript
            meeting_name: Optional name for the meeting
            duration: Optional duration of the audio in seconds
            source: Source of the transcript (e.g., "whisper", "openai_whisper", "anythingllm")
            
        Returns:
            Dictionary with information about the stored transcript
        """
        try:
            # Generate ID if not provided
            transcript_id = metadata.get("id", str(uuid.uuid4()))
            
            # Update metadata
            metadata["id"] = transcript_id
            metadata["created_at"] = metadata.get("created_at", datetime.now().isoformat())
            
            # Add meeting name to metadata if provided
            if meeting_name:
                metadata["meeting_name"] = meeting_name
                
            # Add duration to metadata if provided
            if duration:
                metadata["duration"] = duration
                
            # Add source to metadata
            metadata["source"] = source
            
            # Get embedding for the transcript
            embedding = self._get_embedding(transcript_text)
            
            # Store in LanceDB
            self.transcripts_table.add([{
                "id": transcript_id,
                "content": transcript_text,
                "metadata": json.dumps(metadata),
                "embedding": embedding
            }])
            
            return {
                "success": True,
                "transcript_id": transcript_id,
                "meeting_name": meeting_name,
                "duration": duration,
                "source": source
            }
        except Exception as e:
            logger.error(f"Error storing transcript: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_transcript(self, transcript_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a transcript by ID
        
        Args:
            transcript_id: ID of the transcript
            
        Returns:
            Transcript data or None if not found
        """
        try:
            result = self.transcripts_table.search().where(f"id = '{transcript_id}'").limit(1).to_pandas()
            
            if len(result) == 0:
                return None
            
            row = result.iloc[0]
            metadata = json.loads(row["metadata"])
            
            return {
                "id": row["id"],
                "text": row["content"],
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting transcript {transcript_id}: {str(e)}")
            return None
    
    def get_transcript_text(self, transcript_id: str) -> Optional[str]:
        """
        Get transcript text by ID
        
        Args:
            transcript_id: ID of the transcript
            
        Returns:
            Transcript text or None if not found
        """
        result = self.get_transcript(transcript_id)
        if not result:
            logger.warning(f"Transcript not found with ID: {transcript_id}")
            return None
            
        # Extract the text from the transcript result
        return result.get("text")
    
    def get_all_transcripts(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all transcripts
        
        Returns:
            Dictionary with list of transcripts
        """
        try:
            # Check if the table exists
            if "transcripts" not in self.db.table_names():
                logger.warning("Transcripts table does not exist yet")
                return {"transcripts": []}
                
            # Get all transcripts
            result = self.transcripts_table.search().to_pandas()
            
            if result.empty:
                logger.info("No transcripts found in database")
                return {"transcripts": []}
            
            transcripts = []
            for _, row in result.iterrows():
                try:
                    # Handle potential JSON parsing errors
                    if isinstance(row["metadata"], str):
                        metadata = json.loads(row["metadata"])
                    else:
                        metadata = row["metadata"]
                        
                    transcript = {
                        "id": row["id"],
                        "meeting_name": metadata.get("meeting_name", "Unnamed Meeting"),
                        "created_at": metadata.get("created_at", ""),
                        "duration": metadata.get("duration", 0)
                    }
                    logger.debug(f"Processed transcript: {transcript}")
                    transcripts.append(transcript)
                except Exception as e:
                    logger.error(f"Error processing transcript row {row['id'] if 'id' in row else 'unknown'}: {str(e)}")
                    # Continue processing other rows
            
            logger.info(f"Retrieved {len(transcripts)} transcripts")
            return {"transcripts": transcripts}
        except Exception as e:
            logger.error(f"Error getting all transcripts: {str(e)}")
            return {"transcripts": []}
    
    def delete_transcript(self, transcript_id: str) -> bool:
        """
        Delete a transcript by ID
        
        Args:
            transcript_id: ID of the transcript
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.transcripts_table.delete(f"id = '{transcript_id}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting transcript {transcript_id}: {str(e)}")
            return False
    
    def store_minutes(self, minutes_text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store meeting minutes in LanceDB
        
        Args:
            minutes_text: Minutes text
            metadata: Metadata for the minutes
            
        Returns:
            Dictionary with information about the stored minutes
        """
        try:
            # Generate ID if not provided
            minutes_id = metadata.get("id", str(uuid.uuid4()))
            
            # Update metadata
            metadata["id"] = minutes_id
            metadata["created_at"] = metadata.get("created_at", datetime.now().isoformat())
            
            # Get embedding for the minutes
            embedding = self._get_embedding(minutes_text)
            
            # Store in LanceDB
            self.minutes_table.add([{
                "id": minutes_id,
                "content": minutes_text,
                "metadata": json.dumps(metadata),
                "embedding": embedding
            }])
            
            return {
                "success": True,
                "minutes_id": minutes_id,
            }
        except Exception as e:
            logger.error(f"Error storing minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def store_meeting_minutes(self, transcript_id: str, minutes_text: str, meeting_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store meeting minutes associated with a transcript
        
        Args:
            transcript_id: ID of the associated transcript
            minutes_text: Minutes text
            meeting_name: Optional name for the meeting
            metadata: Optional additional metadata
            
        Returns:
            Dictionary with information about the stored minutes
        """
        try:
            logger.info(f"Storing meeting minutes for transcript {transcript_id}")
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add transcript ID and meeting name to metadata
            metadata["transcript_id"] = transcript_id
            if meeting_name:
                metadata["meeting_name"] = meeting_name
            
            # Use the existing store_minutes method
            result = self.store_minutes(minutes_text, metadata)
            
            logger.info(f"Meeting minutes stored with result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error storing meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_minutes(self, minutes_id: str) -> Optional[Dict[str, Any]]:
        """
        Get meeting minutes by ID
        
        Args:
            minutes_id: ID of the minutes
            
        Returns:
            Minutes data or None if not found
        """
        try:
            result = self.minutes_table.search().where(f"id = '{minutes_id}'").limit(1).to_pandas()
            
            if len(result) == 0:
                return None
            
            row = result.iloc[0]
            metadata = json.loads(row["metadata"])
            
            return {
                "id": row["id"],
                "text": row["content"],
                "metadata": metadata
            }
        except Exception as e:
            logger.error(f"Error getting minutes {minutes_id}: {str(e)}")
            return None
    
    def get_all_minutes(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all meeting minutes
        
        Returns:
            Dictionary with list of minutes
        """
        try:
            result = self.minutes_table.search().to_pandas()
            
            minutes_list = []
            for _, row in result.iterrows():
                metadata = json.loads(row["metadata"])
                minutes_list.append({
                    "id": row["id"],
                    "meeting_name": metadata.get("meeting_name", "Unnamed Meeting"),
                    "created_at": metadata.get("created_at", ""),
                    "transcript_id": metadata.get("transcript_id", "")
                })
            
            return {"minutes": minutes_list}
        except Exception as e:
            logger.error(f"Error getting all minutes: {str(e)}")
            return {"minutes": []}
    
    def delete_minutes(self, minutes_id: str) -> bool:
        """
        Delete meeting minutes by ID
        
        Args:
            minutes_id: ID of the minutes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.minutes_table.delete(f"id = '{minutes_id}'")
            return True
        except Exception as e:
            logger.error(f"Error deleting minutes {minutes_id}: {str(e)}")
            return False
    
    def search_transcripts(self, query: str, top_n: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search transcripts by semantic similarity
        
        Args:
            query: Search query
            top_n: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching transcripts with similarity scores
        """
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search in LanceDB
            results = (
                self.transcripts_table.search(query_embedding)
                .limit(top_n)
                .to_pandas()
            )
            
            # Process results
            matches = []
            for _, row in results.iterrows():
                # Convert distance to similarity score (0-1)
                distance = row["_distance"]
                score = 1.0 - min(distance, 1.0)
                
                if score >= score_threshold:
                    metadata = json.loads(row["metadata"])
                    matches.append({
                        "id": row["id"],
                        "text": row["content"],
                        "metadata": metadata,
                        "score": score
                    })
            
            return matches
        except Exception as e:
            logger.error(f"Error searching transcripts: {str(e)}")
            return []
    
    def search_minutes(self, query: str, top_n: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search meeting minutes by semantic similarity
        
        Args:
            query: Search query
            top_n: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching minutes with similarity scores
        """
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Search in LanceDB
            results = (
                self.minutes_table.search(query_embedding)
                .limit(top_n)
                .to_pandas()
            )
            
            # Process results
            matches = []
            for _, row in results.iterrows():
                # Convert distance to similarity score (0-1)
                distance = row["_distance"]
                score = 1.0 - min(distance, 1.0)
                
                if score >= score_threshold:
                    metadata = json.loads(row["metadata"])
                    matches.append({
                        "id": row["id"],
                        "text": row["content"],
                        "metadata": metadata,
                        "score": score
                    })
            
            return matches
        except Exception as e:
            logger.error(f"Error searching minutes: {str(e)}")
            return []
    
    def get_relevant_context(self, query: str, max_tokens: int = 3000) -> str:
        """
        Get relevant context for a query by searching both transcripts and minutes
        
        Args:
            query: Search query
            max_tokens: Maximum number of tokens to return
            
        Returns:
            Relevant context as a string
        """
        # Search both transcripts and minutes
        transcript_results = self.search_transcripts(query, top_n=3, score_threshold=0.6)
        minutes_results = self.search_minutes(query, top_n=2, score_threshold=0.6)
        
        # Combine results
        all_results = transcript_results + minutes_results
        
        # Sort by score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Build context string
        context_parts = []
        token_count = 0
        
        for result in all_results:
            # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
            text = result["text"]
            estimated_tokens = len(text) // 4
            
            if token_count + estimated_tokens > max_tokens:
                # Truncate text to fit within token limit
                available_tokens = max_tokens - token_count
                available_chars = available_tokens * 4
                text = text[:available_chars] + "..."
            
            # Add source information
            result_type = "Transcript" if result["id"] in [r["id"] for r in transcript_results] else "Minutes"
            meeting_name = result["metadata"].get("meeting_name", "Unnamed Meeting")
            
            context_parts.append(f"--- {result_type}: {meeting_name} ---\n{text}\n")
            
            token_count += estimated_tokens
            if token_count >= max_tokens:
                break
        
        return "\n".join(context_parts)
