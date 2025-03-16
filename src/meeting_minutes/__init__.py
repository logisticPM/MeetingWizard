"""
Meeting Minutes Package

This package provides functionality for generating and storing meeting minutes
from transcribed audio.
"""

from .meeting_summarizer import MeetingSummarizer
from .transcript_storage import TranscriptStorage

__all__ = ['MeetingSummarizer', 'TranscriptStorage']
