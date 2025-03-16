"""
Transcript Storage Module

This module provides functionality to store and retrieve meeting transcripts
using a local SQLite database and file system.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptStorage:
    """Class to handle storage and retrieval of meeting transcripts"""
    
    def __init__(self, db_path="data/meetings/transcripts.db", storage_dir="data/meetings/transcripts"):
        """
        Initialize the transcript storage
        
        Args:
            db_path: Path to the SQLite database
            storage_dir: Directory to store transcript files
        """
        self.db_path = db_path
        self.storage_dir = storage_dir
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Initialize the database
        self._init_db()
    
    def _init_db(self):
        """Initialize the database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id TEXT PRIMARY KEY,
                    meeting_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    duration REAL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meeting_minutes (
                    id TEXT PRIMARY KEY,
                    transcript_id TEXT NOT NULL,
                    meeting_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    stored_in_anything_llm BOOLEAN DEFAULT 0,
                    metadata TEXT,
                    FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transcript_chunks (
                    id TEXT PRIMARY KEY,
                    transcript_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
                )
            """)
            
            conn.commit()
    
    def store_transcript(self, 
                         transcript_text: str, 
                         meeting_name: Optional[str] = None, 
                         duration: Optional[float] = None, 
                         source: str = "whisper", 
                         metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a transcript in the database and file system
        
        Args:
            transcript_text: The transcript text
            meeting_name: Name of the meeting
            duration: Duration of the audio in seconds
            source: Source of the transcript (e.g., "whisper", "anythingllm")
            metadata: Additional metadata
        
        Returns:
            Dictionary with information about the stored transcript
        """
        try:
            # Generate a unique ID
            transcript_id = str(uuid.uuid4())
            
            # Generate a meeting name if not provided
            if not meeting_name:
                meeting_name = f"Meeting Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            
            # Create a file path for the transcript
            file_name = f"{transcript_id}.txt"
            file_path = os.path.join(self.storage_dir, file_name)
            
            # Save the transcript to a file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            
            # Save metadata to the database
            created_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO transcripts (id, meeting_name, file_path, duration, source, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        transcript_id,
                        meeting_name,
                        file_path,
                        duration,
                        source,
                        created_at,
                        json.dumps(metadata) if metadata else None
                    )
                )
                conn.commit()
            
            # Create chunks for easier processing
            self._create_transcript_chunks(transcript_id, transcript_text)
            
            logger.info(f"Stored transcript: {meeting_name} (ID: {transcript_id})")
            
            return {
                "success": True,
                "transcript_id": transcript_id,
                "meeting_name": meeting_name,
                "file_path": file_path,
                "created_at": created_at
            }
        
        except Exception as e:
            logger.error(f"Failed to store transcript: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_transcript_chunks(self, transcript_id: str, transcript_text: str, chunk_size: int = 1000):
        """
        Create chunks from the transcript for easier processing
        
        Args:
            transcript_id: ID of the transcript
            transcript_text: The transcript text
            chunk_size: Size of each chunk in characters
        """
        # Split the transcript into chunks
        chunks = []
        for i in range(0, len(transcript_text), chunk_size):
            chunk_text = transcript_text[i:i+chunk_size]
            chunks.append((str(uuid.uuid4()), transcript_id, len(chunks), chunk_text))
        
        # Store the chunks in the database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO transcript_chunks (id, transcript_id, chunk_index, chunk_text)
                VALUES (?, ?, ?, ?)
                """,
                chunks
            )
            conn.commit()
    
    def get_transcript(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get a transcript by ID
        
        Args:
            transcript_id: ID of the transcript
        
        Returns:
            Dictionary with transcript information and text
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM transcripts WHERE id = ?
                    """,
                    (transcript_id,)
                )
                transcript = cursor.fetchone()
                
                if not transcript:
                    return {
                        "success": False,
                        "error": f"Transcript not found: {transcript_id}"
                    }
                
                # Read the transcript text from the file
                with open(transcript["file_path"], "r", encoding="utf-8") as f:
                    transcript_text = f.read()
                
                return {
                    "success": True,
                    "transcript": {
                        "id": transcript["id"],
                        "meeting_name": transcript["meeting_name"],
                        "file_path": transcript["file_path"],
                        "duration": transcript["duration"],
                        "source": transcript["source"],
                        "created_at": transcript["created_at"],
                        "metadata": json.loads(transcript["metadata"]) if transcript["metadata"] else None,
                        "text": transcript_text
                    }
                }
        
        except Exception as e:
            logger.error(f"Failed to get transcript: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_transcripts(self) -> Dict[str, Any]:
        """
        Get all transcripts
        
        Returns:
            Dictionary with list of transcripts
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, meeting_name, duration, source, created_at FROM transcripts
                    ORDER BY created_at DESC
                    """
                )
                transcripts = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "success": True,
                    "transcripts": transcripts
                }
        
        except Exception as e:
            logger.error(f"Failed to get transcripts: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_transcript(self, transcript_id: str) -> Dict[str, Any]:
        """
        Delete a transcript
        
        Args:
            transcript_id: ID of the transcript
        
        Returns:
            Dictionary with result of the operation
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the file path
                cursor.execute(
                    """
                    SELECT file_path FROM transcripts WHERE id = ?
                    """,
                    (transcript_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return {
                        "success": False,
                        "error": f"Transcript not found: {transcript_id}"
                    }
                
                file_path = result[0]
                
                # Delete related meeting minutes
                cursor.execute(
                    """
                    DELETE FROM meeting_minutes WHERE transcript_id = ?
                    """,
                    (transcript_id,)
                )
                
                # Delete transcript chunks
                cursor.execute(
                    """
                    DELETE FROM transcript_chunks WHERE transcript_id = ?
                    """,
                    (transcript_id,)
                )
                
                # Delete the transcript
                cursor.execute(
                    """
                    DELETE FROM transcripts WHERE id = ?
                    """,
                    (transcript_id,)
                )
                
                conn.commit()
                
                # Delete the file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return {
                    "success": True,
                    "message": f"Transcript deleted: {transcript_id}"
                }
        
        except Exception as e:
            logger.error(f"Failed to delete transcript: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def store_meeting_minutes(self, 
                             transcript_id: str, 
                             minutes_text: str, 
                             meeting_name: Optional[str] = None,
                             stored_in_anything_llm: bool = False,
                             metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store meeting minutes associated with a transcript
        
        Args:
            transcript_id: ID of the associated transcript
            minutes_text: The meeting minutes text
            meeting_name: Name of the meeting
            stored_in_anything_llm: Whether the minutes are stored in AnythingLLM
            metadata: Additional metadata
        
        Returns:
            Dictionary with information about the stored minutes
        """
        try:
            # Check if the transcript exists
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT meeting_name FROM transcripts WHERE id = ?
                    """,
                    (transcript_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return {
                        "success": False,
                        "error": f"Transcript not found: {transcript_id}"
                    }
                
                # Use the transcript's meeting name if not provided
                if not meeting_name:
                    meeting_name = result[0]
            
            # Generate a unique ID
            minutes_id = str(uuid.uuid4())
            
            # Create a file path for the minutes
            file_name = f"{minutes_id}.md"
            file_path = os.path.join(self.storage_dir, file_name)
            
            # Save the minutes to a file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(minutes_text)
            
            # Save metadata to the database
            created_at = datetime.now().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO meeting_minutes (id, transcript_id, meeting_name, file_path, created_at, stored_in_anything_llm, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        minutes_id,
                        transcript_id,
                        meeting_name,
                        file_path,
                        created_at,
                        1 if stored_in_anything_llm else 0,
                        json.dumps(metadata) if metadata else None
                    )
                )
                conn.commit()
            
            logger.info(f"Stored meeting minutes: {meeting_name} (ID: {minutes_id})")
            
            return {
                "success": True,
                "minutes_id": minutes_id,
                "transcript_id": transcript_id,
                "meeting_name": meeting_name,
                "file_path": file_path,
                "created_at": created_at
            }
        
        except Exception as e:
            logger.error(f"Failed to store meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_meeting_minutes(self, minutes_id: str) -> Dict[str, Any]:
        """
        Get meeting minutes by ID
        
        Args:
            minutes_id: ID of the meeting minutes
        
        Returns:
            Dictionary with meeting minutes information and text
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM meeting_minutes WHERE id = ?
                    """,
                    (minutes_id,)
                )
                minutes = cursor.fetchone()
                
                if not minutes:
                    return {
                        "success": False,
                        "error": f"Meeting minutes not found: {minutes_id}"
                    }
                
                # Read the minutes text from the file
                with open(minutes["file_path"], "r", encoding="utf-8") as f:
                    minutes_text = f.read()
                
                return {
                    "success": True,
                    "minutes": {
                        "id": minutes["id"],
                        "transcript_id": minutes["transcript_id"],
                        "meeting_name": minutes["meeting_name"],
                        "file_path": minutes["file_path"],
                        "created_at": minutes["created_at"],
                        "stored_in_anything_llm": bool(minutes["stored_in_anything_llm"]),
                        "metadata": json.loads(minutes["metadata"]) if minutes["metadata"] else None,
                        "text": minutes_text
                    }
                }
        
        except Exception as e:
            logger.error(f"Failed to get meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_all_meeting_minutes(self) -> Dict[str, Any]:
        """
        Get all meeting minutes
        
        Returns:
            Dictionary with list of meeting minutes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, transcript_id, meeting_name, created_at, stored_in_anything_llm 
                    FROM meeting_minutes
                    ORDER BY created_at DESC
                    """
                )
                minutes_list = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "success": True,
                    "meeting_minutes": minutes_list
                }
        
        except Exception as e:
            logger.error(f"Failed to get meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_transcript_minutes(self, transcript_id: str) -> Dict[str, Any]:
        """
        Get all meeting minutes associated with a transcript
        
        Args:
            transcript_id: ID of the transcript
        
        Returns:
            Dictionary with list of meeting minutes
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT id, meeting_name, created_at, stored_in_anything_llm 
                    FROM meeting_minutes
                    WHERE transcript_id = ?
                    ORDER BY created_at DESC
                    """,
                    (transcript_id,)
                )
                minutes_list = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "success": True,
                    "transcript_id": transcript_id,
                    "meeting_minutes": minutes_list
                }
        
        except Exception as e:
            logger.error(f"Failed to get transcript meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_meeting_minutes(self, minutes_id: str) -> Dict[str, Any]:
        """
        Delete meeting minutes
        
        Args:
            minutes_id: ID of the meeting minutes
        
        Returns:
            Dictionary with result of the operation
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get the file path
                cursor.execute(
                    """
                    SELECT file_path FROM meeting_minutes WHERE id = ?
                    """,
                    (minutes_id,)
                )
                result = cursor.fetchone()
                
                if not result:
                    return {
                        "success": False,
                        "error": f"Meeting minutes not found: {minutes_id}"
                    }
                
                file_path = result[0]
                
                # Delete the meeting minutes
                cursor.execute(
                    """
                    DELETE FROM meeting_minutes WHERE id = ?
                    """,
                    (minutes_id,)
                )
                
                conn.commit()
                
                # Delete the file
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                return {
                    "success": True,
                    "message": f"Meeting minutes deleted: {minutes_id}"
                }
        
        except Exception as e:
            logger.error(f"Failed to delete meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_transcripts(self, query: str) -> Dict[str, Any]:
        """
        Search transcripts for a query string
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with search results
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Search in transcript chunks
                cursor.execute(
                    """
                    SELECT tc.transcript_id, tc.chunk_text, t.meeting_name, t.created_at
                    FROM transcript_chunks tc
                    JOIN transcripts t ON tc.transcript_id = t.id
                    WHERE tc.chunk_text LIKE ?
                    ORDER BY t.created_at DESC
                    LIMIT 20
                    """,
                    (f"%{query}%",)
                )
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "transcript_id": row["transcript_id"],
                        "meeting_name": row["meeting_name"],
                        "created_at": row["created_at"],
                        "text_snippet": row["chunk_text"][:200] + "..." if len(row["chunk_text"]) > 200 else row["chunk_text"]
                    })
                
                return {
                    "success": True,
                    "results": results
                }
        
        except Exception as e:
            logger.error(f"Failed to search transcripts: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_meeting_minutes(self, query: str) -> Dict[str, Any]:
        """
        Search meeting minutes for a query string
        
        Args:
            query: Search query
        
        Returns:
            Dictionary with search results
        """
        try:
            # Get all meeting minutes
            minutes_result = self.get_all_meeting_minutes()
            
            if not minutes_result["success"]:
                return minutes_result
            
            results = []
            for minutes in minutes_result["meeting_minutes"]:
                # Get the full text
                full_minutes = self.get_meeting_minutes(minutes["id"])
                
                if full_minutes["success"] and query.lower() in full_minutes["minutes"]["text"].lower():
                    # Find the context around the match
                    text = full_minutes["minutes"]["text"]
                    query_pos = text.lower().find(query.lower())
                    start_pos = max(0, query_pos - 100)
                    end_pos = min(len(text), query_pos + len(query) + 100)
                    
                    snippet = "..." if start_pos > 0 else ""
                    snippet += text[start_pos:end_pos]
                    snippet += "..." if end_pos < len(text) else ""
                    
                    results.append({
                        "minutes_id": minutes["id"],
                        "transcript_id": minutes["transcript_id"],
                        "meeting_name": minutes["meeting_name"],
                        "created_at": minutes["created_at"],
                        "text_snippet": snippet
                    })
            
            return {
                "success": True,
                "results": results
            }
        
        except Exception as e:
            logger.error(f"Failed to search meeting minutes: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
