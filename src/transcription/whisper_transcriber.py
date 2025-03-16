"""
OpenAI Whisper Transcription Service for simple-npu-chatbot
"""

import os
import logging
import tempfile
import json
from typing import Dict, Any, Optional
import openai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionSettings:
    """Class to manage transcription settings"""
    
    def __init__(self, settings_file="data/transcription_settings.json"):
        """
        Initialize transcription settings
        
        Args:
            settings_file: Path to settings file
        """
        self.settings_file = settings_file
        self.api_key = None
        self.model = "whisper-1"
        self.language = None
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        
        # Load settings if file exists
        self.load_settings()
    
    def load_settings(self):
        """Load settings from file"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.api_key = settings.get("api_key")
                    self.model = settings.get("model", "whisper-1")
                    self.language = settings.get("language")
            except Exception as e:
                logger.error(f"Failed to load settings: {str(e)}")
    
    def save_settings(self, api_key, model="whisper-1", language=None):
        """
        Save settings to file
        
        Args:
            api_key: OpenAI API key
            model: Whisper model name
            language: Language code
        
        Returns:
            Dictionary with result
        """
        try:
            settings = {
                "api_key": api_key,
                "model": model,
                "language": language
            }
            
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
            
            self.api_key = api_key
            self.model = model
            self.language = language
            
            return {
                "success": True,
                "message": "Settings saved successfully"
            }
        except Exception as e:
            logger.error(f"Failed to save settings: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to save settings: {str(e)}"
            }
    
    def is_available(self):
        """Check if API key is set"""
        return bool(self.api_key)

class OpenAIWhisperService:
    """Class to handle transcription using OpenAI Whisper API"""
    
    def __init__(self, api_key=None, model="whisper-1", language=None):
        """
        Initialize the OpenAI Whisper service
        
        Args:
            api_key: OpenAI API key
            model: Whisper model name
            language: Language code
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        self.client = None
        
        # Initialize the OpenAI client if API key is provided
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
    
    def transcribe(self, audio_file_path):
        """
        Transcribe audio file using OpenAI Whisper API
        
        Args:
            audio_file_path: Path to audio file
        
        Returns:
            Dictionary with transcription result
        """
        if not self.api_key:
            return {
                "success": False,
                "error": "API key not set",
                "needs_api_key": True
            }
        
        try:
            # Ensure client is initialized
            if not self.client:
                try:
                    from openai import OpenAI
                    self.client = OpenAI(api_key=self.api_key)
                except Exception as e:
                    logger.error(f"Error initializing OpenAI client: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Failed to initialize OpenAI client: {str(e)}"
                    }
            
            # Prepare transcription options
            options = {
                "model": self.model
            }
            
            # Add language if specified
            if self.language:
                options["language"] = self.language
            
            # Open the audio file
            with open(audio_file_path, "rb") as audio_file:
                # Call the OpenAI API using the client instance
                response = self.client.audio.transcriptions.create(
                    file=audio_file,
                    **options
                )
            
            # Get the transcription text
            transcription = response.text
            
            # Get audio duration if possible
            duration = None
            try:
                # First try to get duration using wave module for WAV files
                if audio_file_path.lower().endswith('.wav'):
                    import wave
                    with wave.open(audio_file_path, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                        rate = wav_file.getframerate()
                        duration = frames / float(rate)
                else:
                    # For other audio formats (mp3, m4a, etc.), use mutagen
                    try:
                        import mutagen
                        audio = mutagen.File(audio_file_path)
                        if audio is not None:
                            duration = audio.info.length
                        else:
                            # Fallback to pydub if mutagen fails
                            try:
                                from pydub import AudioSegment
                                audio_format = audio_file_path.split('.')[-1].lower()
                                audio = AudioSegment.from_file(audio_file_path, format=audio_format)
                                duration = len(audio) / 1000.0  # pydub duration is in milliseconds
                            except ImportError:
                                logger.warning("Neither mutagen nor pydub is installed. Cannot determine audio duration for non-WAV files.")
                    except ImportError:
                        logger.warning("Mutagen is not installed. Cannot determine audio duration for non-WAV files.")
            except Exception as e:
                logger.warning(f"Could not determine audio duration: {str(e)}")
            
            return {
                "success": True,
                "text": transcription,
                "duration": duration,
                "model": self.model,
                "language": self.language
            }
        
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return {
                "success": False,
                "error": f"Transcription failed: {str(e)}"
            }
    
    def transcribe_and_store(self, audio_file_path, transcript_storage, meeting_name=None):
        """
        Transcribe audio file and store the transcript
        
        Args:
            audio_file_path: Path to audio file
            transcript_storage: TranscriptStorage instance
            meeting_name: Optional name for the meeting
        
        Returns:
            Dictionary with transcription and storage result
        """
        # Transcribe the audio
        result = self.transcribe(audio_file_path)
        
        if not result.get("success"):
            return result
        
        # Store the transcript
        metadata = {
            "model": self.model,
            "language": self.language,
            "source": "openai_whisper"
        }
        
        storage_result = transcript_storage.store_transcript(
            transcript_text=result.get("text"),
            meeting_name=meeting_name,
            duration=result.get("duration"),
            source="openai_whisper",
            metadata=metadata
        )
        
        if not storage_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to store transcript: {storage_result.get('error')}",
                "transcription_success": True,
                "transcription": result.get("text")
            }
        
        # Combine the results
        return {
            "success": True,
            "text": result.get("text"),
            "transcript_id": storage_result.get("transcript_id"),
            "meeting_name": storage_result.get("meeting_name"),
            "duration": result.get("duration"),
            "model": self.model,
            "language": self.language
        }
