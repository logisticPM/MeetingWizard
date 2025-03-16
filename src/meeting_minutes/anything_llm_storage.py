"""
AnythingLLM Storage Module for Meeting Minutes

This module provides functions to store and retrieve meeting minutes using AnythingLLM's
document management and RAG capabilities.
"""

import os
import json
import requests
import yaml
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnythingLLMStorage:
    """Class to handle storage and retrieval of meeting minutes using AnythingLLM"""
    
    def __init__(self, config_path="config.yaml"):
        """
        Initialize the AnythingLLM storage manager
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        
        self.api_key = config["api_key"]
        self.base_url = config["model_server_base_url"]
        self.workspace_slug = config["workspace_slug"]
        
        # API endpoints
        self.documents_url = f"{self.base_url}/workspaces/{self.workspace_slug}/documents"
        self.chat_url = f"{self.base_url}/workspace/{self.workspace_slug}/chat"
        
        # Default headers
        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def store_meeting_minutes(self, transcript: str, summary: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Store meeting minutes in AnythingLLM as a document
        
        Args:
            transcript: The meeting transcript
            summary: The meeting summary
            meeting_name: Optional name for the meeting
        
        Returns:
            Response from the API
        """
        # Generate meeting name if not provided
        if not meeting_name:
            meeting_name = f"Meeting Minutes - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Create a document with both transcript and summary
        document_content = f"# {meeting_name}\n\n## Summary\n\n{summary}\n\n## Full Transcript\n\n{transcript}"
        
        # Save to a temporary file
        temp_file_path = f"temp_meeting_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(document_content)
        
        try:
            # Upload the document to AnythingLLM
            with open(temp_file_path, "rb") as f:
                files = {
                    "file": (f"{meeting_name}.md", f, "text/markdown")
                }
                
                upload_headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                
                response = requests.post(
                    self.documents_url,
                    headers=upload_headers,
                    files=files
                )
                
                response.raise_for_status()
                result = response.json()
                
                logger.info(f"Successfully stored meeting minutes: {meeting_name}")
                return {
                    "success": True,
                    "message": "Meeting minutes stored successfully",
                    "data": result
                }
        
        except Exception as e:
            logger.error(f"Failed to store meeting minutes: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to store meeting minutes: {str(e)}"
            }
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def query_meeting_minutes(self, query: str) -> Dict[str, Any]:
        """
        Query the stored meeting minutes using AnythingLLM's RAG capabilities
        
        Args:
            query: The query to search for in the meeting minutes
        
        Returns:
            Response from the API with relevant information
        """
        try:
            data = {
                "message": query,
                "mode": "retrieval",
                "sessionId": f"meeting-minutes-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "attachments": []
            }
            
            response = requests.post(
                self.chat_url,
                headers=self.headers,
                json=data
            )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "success": True,
                "message": "Query successful",
                "data": result
            }
        
        except Exception as e:
            logger.error(f"Failed to query meeting minutes: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to query meeting minutes: {str(e)}"
            }
    
    def get_all_meeting_documents(self) -> Dict[str, Any]:
        """
        Get a list of all meeting minutes documents stored in AnythingLLM
        
        Returns:
            Response from the API with the list of documents
        """
        try:
            response = requests.get(
                self.documents_url,
                headers=self.headers
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Filter for meeting minutes documents
            meeting_docs = [doc for doc in result if "Meeting Minutes" in doc.get("name", "")]
            
            return {
                "success": True,
                "message": "Retrieved meeting documents successfully",
                "data": meeting_docs
            }
        
        except Exception as e:
            logger.error(f"Failed to get meeting documents: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to get meeting documents: {str(e)}"
            }
