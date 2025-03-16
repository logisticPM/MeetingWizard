"""
AnythingLLM Client Module

This module provides a client for interacting with the AnythingLLM API.
"""

import requests
import logging
import json
import os
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnythingLLMClient:
    """Client for interacting with the AnythingLLM API"""
    
    def __init__(self, base_url="http://localhost:3001", api_key=None, workspace_slug=None):
        """
        Initialize the AnythingLLM client
        
        Args:
            base_url: Base URL for the AnythingLLM API
            api_key: API key for authentication
            workspace_slug: Workspace slug to use
        """
        self.base_url = base_url
        self.api_key = api_key
        self.workspace_slug = workspace_slug
        
        # Load config if available
        self._load_config()
    
    def _load_config(self):
        """Load configuration from config file if it exists"""
        try:
            if os.path.exists("config.yaml"):
                import yaml
                with open("config.yaml", "r") as file:
                    config = yaml.safe_load(file)
                
                # Only override if not already set
                if not self.api_key and "api_key" in config:
                    self.api_key = config["api_key"]
                
                if not self.base_url and "model_server_base_url" in config:
                    self.base_url = config["model_server_base_url"]
                
                if not self.workspace_slug and "workspace_slug" in config:
                    self.workspace_slug = config["workspace_slug"]
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
    
    def _get_headers(self):
        """Get headers for API requests"""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def get_headers(self):
        """Get headers for API requests"""
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def chat(self, message: str, mode="chat", session_id=None) -> Dict[str, Any]:
        """
        Send a chat request to the AnythingLLM API
        
        Args:
            message: The message to send
            mode: Chat mode (chat or retrieval)
            session_id: Session ID for the chat
        
        Returns:
            Response from the API
        """
        if not self.workspace_slug:
            return {
                "success": False,
                "error": "Workspace slug not set"
            }
        
        if not session_id:
            import uuid
            session_id = f"session-{uuid.uuid4()}"
        
        url = f"{self.base_url}/workspace/{self.workspace_slug}/chat"
        
        data = {
            "message": message,
            "mode": mode,
            "sessionId": session_id,
            "attachments": []
        }
        
        try:
            response = requests.post(
                url,
                headers=self.get_headers(),
                json=data
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Chat request failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_workspaces(self) -> Dict[str, Any]:
        """
        Get a list of available workspaces
        
        Returns:
            Response from the API
        """
        url = f"{self.base_url}/workspaces"
        
        try:
            response = requests.get(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get workspaces: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_documents(self) -> Dict[str, Any]:
        """
        Get a list of documents in the workspace
        
        Returns:
            Response from the API
        """
        if not self.workspace_slug:
            return {
                "success": False,
                "error": "Workspace slug not set"
            }
        
        url = f"{self.base_url}/workspaces/{self.workspace_slug}/documents"
        
        try:
            response = requests.get(
                url,
                headers=self.get_headers()
            )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get documents: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_document(self, file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a document to the workspace
        
        Args:
            file_path: Path to the file to upload
            file_name: Optional name for the file
        
        Returns:
            Response from the API
        """
        if not self.workspace_slug:
            return {
                "success": False,
                "error": "Workspace slug not set"
            }
        
        url = f"{self.base_url}/workspaces/{self.workspace_slug}/documents"
        
        if not file_name:
            import os
            file_name = os.path.basename(file_path)
        
        try:
            with open(file_path, "rb") as f:
                files = {
                    "file": (file_name, f)
                }
                
                response = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}" if self.api_key else ""},
                    files=files
                )
                
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to upload document: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using AnythingLLM's NPU Embedder
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Use the OpenAI compatible embeddings endpoint
            url = f"{self.base_url}/api/v1/openai/embeddings"
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                json={"inputs": [text], "model": None}
            )
            
            if response.status_code == 200:
                embedding_data = response.json()
                if "data" in embedding_data and len(embedding_data["data"]) > 0:
                    return embedding_data["data"][0]["embedding"]
                else:
                    logger.warning("AnythingLLM embedding response missing 'data' field or empty data")
            else:
                logger.warning(f"AnythingLLM embedding request failed: {response.status_code} - {response.text}")
            
            return None
        except Exception as e:
            logger.error(f"Error getting embedding from AnythingLLM: {str(e)}")
            return None
    
    def check_connection(self) -> bool:
        """
        Check if the connection to AnythingLLM is working
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                headers=self.get_headers()
            )
            
            return response.status_code == 200
        except Exception:
            return False
