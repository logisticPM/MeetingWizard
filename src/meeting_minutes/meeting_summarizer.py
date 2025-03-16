"""
Meeting Summarizer Module

This module provides functionality to generate meeting minutes and summaries
from transcribed audio using OpenAI's API.
"""

import os
import json
import logging
import openai
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeetingSummarizer:
    """Class to generate meeting summaries and minutes using OpenAI"""
    
    def __init__(self, api_key=None, model="gpt-3.5-turbo"):
        """
        Initialize the meeting summarizer
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use for summarization
        """
        self.api_key = api_key
        self.model = model
        
        # If no API key is provided, try to get it from environment variables
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")
        
        # Set the API key for OpenAI
        if self.api_key:
            openai.api_key = self.api_key
    
    def is_available(self):
        """Check if the summarizer is available (has API key)"""
        return bool(self.api_key)
    
    def generate_summary_from_transcript_id(self, transcript_id: str, transcript_storage, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a meeting summary from a stored transcript ID
        
        Args:
            transcript_id: The ID of the stored transcript
            transcript_storage: TranscriptStorage instance
            meeting_name: Optional name for the meeting
        
        Returns:
            Dictionary containing the summary and structured information
        """
        # Get the transcript from storage
        transcript_data = transcript_storage.get_transcript(transcript_id)
        
        if not transcript_data:
            logger.error(f"Failed to retrieve transcript with ID: {transcript_id}")
            return {
                "success": False,
                "error": f"Failed to retrieve transcript with ID: {transcript_id}"
            }
        
        transcript_text = transcript_data.get("text")
        
        if not transcript_text:
            logger.error(f"No text found in transcript with ID: {transcript_id}")
            return {
                "success": False,
                "error": "No text found in transcript"
            }
        
        # Get metadata from transcript
        metadata = transcript_data.get("metadata", {})
        
        # Use the transcript's meeting name if not provided
        if not meeting_name:
            meeting_name = metadata.get("meeting_name")
        
        logger.info(f"Generating summary for transcript: {transcript_id}, meeting: {meeting_name}")
        
        # Generate summary
        summary_result = self.generate_summary(transcript_text, meeting_name)
        
        if not summary_result.get("success"):
            return summary_result
        
        # Store the meeting minutes
        minutes_text = self.format_meeting_minutes(summary_result)
        
        storage_result = transcript_storage.store_meeting_minutes(
            transcript_id=transcript_id,
            minutes_text=minutes_text,
            meeting_name=meeting_name,
            metadata=summary_result.get("structure")
        )
        
        if not storage_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to store meeting minutes: {storage_result.get('error')}",
                "summary_success": True,
                "summary": summary_result
            }
        
        # Return combined results
        return {
            "success": True,
            "summary": summary_result.get("summary"),
            "structure": summary_result.get("structure"),
            "minutes_id": storage_result.get("minutes_id"),
            "transcript_id": transcript_id,
            "meeting_name": meeting_name,
            "formatted_minutes": minutes_text
        }
    
    def generate_summary(self, transcript: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a meeting summary from a transcript
        
        Args:
            transcript: The meeting transcript
            meeting_name: Optional name for the meeting
        
        Returns:
            Dictionary containing the summary and structured information
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenAI API key not set. Please configure it in the settings tab.",
                "needs_api_key": True
            }
        
        try:
            # Check if we should use AnythingLLM for processing
            use_anything_llm = self._check_anything_llm_available()
            
            if use_anything_llm:
                # Use AnythingLLM's OpenAI-compatible endpoint for RAG
                return self._generate_summary_with_anything_llm(transcript, meeting_name)
            else:
                # Use OpenAI with chunking for long transcripts
                return self._generate_summary_with_openai_chunking(transcript, meeting_name)
                
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating summary: {str(e)}"
            }
    
    def _check_anything_llm_available(self) -> bool:
        """Check if AnythingLLM is available for processing"""
        try:
            from src.anything_llm.anything_llm_client import AnythingLLMClient
            import os
            import json
            
            # Try to get AnythingLLM settings from environment or settings file
            anything_llm_url = os.environ.get("ANYTHING_LLM_URL", "")
            anything_llm_api_key = os.environ.get("ANYTHING_LLM_API_KEY", "")
            
            # If not in environment, try to load from settings file
            if not anything_llm_url or not anything_llm_api_key:
                try:
                    if os.path.exists("settings.json"):
                        with open("settings.json", "r") as f:
                            settings = json.load(f)
                            anything_llm_url = settings.get("anything_llm_url", anything_llm_url)
                            anything_llm_api_key = settings.get("anything_llm_api_key", anything_llm_api_key)
                except Exception as e:
                    logger.warning(f"Could not load AnythingLLM settings: {str(e)}")
            
            # Check if AnythingLLM is configured
            if anything_llm_url:
                client = AnythingLLMClient(
                    base_url=anything_llm_url,
                    api_key=anything_llm_api_key
                )
                
                # Check if connection is working
                return client.check_connection()
            
            return False
        except Exception as e:
            logger.warning(f"Error checking AnythingLLM availability: {str(e)}")
            return False
    
    def _generate_summary_with_anything_llm(self, transcript: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a meeting summary using AnythingLLM's OpenAI-compatible endpoint
        
        Args:
            transcript: The meeting transcript
            meeting_name: Optional name for the meeting
            
        Returns:
            Dictionary containing the summary and structured information
        """
        try:
            from src.anything_llm.anything_llm_client import AnythingLLMClient
            import os
            import json
            
            # Get AnythingLLM settings
            anything_llm_url = os.environ.get("ANYTHING_LLM_URL", "http://localhost:3001")
            anything_llm_api_key = os.environ.get("ANYTHING_LLM_API_KEY", "")
            anything_llm_workspace = os.environ.get("ANYTHING_LLM_WORKSPACE", "default")
            
            # If not in environment, try to load from settings file
            try:
                if os.path.exists("settings.json"):
                    with open("settings.json", "r") as f:
                        settings = json.load(f)
                        anything_llm_url = settings.get("anything_llm_url", anything_llm_url)
                        anything_llm_api_key = settings.get("anything_llm_api_key", anything_llm_api_key)
                        anything_llm_workspace = settings.get("anything_llm_workspace", anything_llm_workspace)
            except Exception as e:
                logger.warning(f"Could not load AnythingLLM settings: {str(e)}")
            
            # Initialize AnythingLLM client
            client = AnythingLLMClient(
                base_url=anything_llm_url,
                api_key=anything_llm_api_key,
                workspace_slug=anything_llm_workspace
            )
            
            # Create system prompts
            system_prompt = """
            You are a professional meeting summarizer. Your task is to create a comprehensive summary of the meeting transcript provided.
            
            Please structure your summary as follows:
            
            1. Meeting Overview: A brief 2-3 sentence overview of what the meeting was about.
            
            2. Key Points: Bullet points of the main topics discussed.
            
            3. Action Items: List any tasks, assignments, or follow-up actions mentioned in the meeting, including who is responsible and any deadlines.
            
            4. Decisions Made: List any decisions that were finalized during the meeting.
            
            5. Next Steps: What are the immediate next steps discussed in the meeting.
            
            Keep your summary concise but comprehensive, focusing on the most important information.
            """
            
            structure_prompt = """
            Based on the meeting transcript, extract the following structured information in JSON format:
            
            {
                "meeting_name": "Auto-generated name if not provided",
                "key_points": [
                    {"point": "Key point 1"},
                    {"point": "Key point 2"}
                ],
                "action_items": [
                    {"item": "Action item 1", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"},
                    {"item": "Action item 2", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"}
                ],
                "decisions": [
                    {"decision": "Decision 1"},
                    {"decision": "Decision 2"}
                ],
                "next_steps": [
                    {"step": "Next step 1"},
                    {"step": "Next step 2"}
                ]
            }
            
            Only include the JSON in your response, nothing else.
            """
            
            # Use AnythingLLM's OpenAI-compatible chat completions endpoint
            # This will automatically handle chunking and RAG for long transcripts
            summary_response = client.chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Here is the meeting transcript:\n\n{transcript}"}
                ],
                model=anything_llm_workspace,
                temperature=0.3
            )
            
            if not summary_response.get("success", False):
                logger.warning(f"Error from AnythingLLM: {summary_response.get('error')}")
                # Fall back to OpenAI with chunking
                return self._generate_summary_with_openai_chunking(transcript, meeting_name)
            
            summary = summary_response.get("response", "")
            
            # Get structured data
            structure_response = client.chat_completion(
                messages=[
                    {"role": "system", "content": structure_prompt},
                    {"role": "user", "content": f"Here is the meeting transcript:\n\n{transcript}"}
                ],
                model=anything_llm_workspace,
                temperature=0.3
            )
            
            if not structure_response.get("success", False):
                logger.warning(f"Error getting structure from AnythingLLM: {structure_response.get('error')}")
                # Create a basic structure
                structure = {
                    "meeting_name": meeting_name or "Meeting Summary",
                    "key_points": [],
                    "action_items": [],
                    "decisions": [],
                    "next_steps": []
                }
            else:
                structure_text = structure_response.get("response", "")
                
                # Parse the JSON structure
                try:
                    # Find JSON in the response
                    import re
                    json_match = re.search(r'({.*})', structure_text, re.DOTALL)
                    if json_match:
                        structure_text = json_match.group(1)
                    
                    structure = json.loads(structure_text)
                    
                    # Use provided meeting name if available
                    if meeting_name:
                        structure["meeting_name"] = meeting_name
                except Exception as e:
                    logger.error(f"Error parsing structure JSON: {str(e)}")
                    # Create a basic structure
                    structure = {
                        "meeting_name": meeting_name or "Meeting Summary",
                        "key_points": [],
                        "action_items": [],
                        "decisions": [],
                        "next_steps": []
                    }
            
            return {
                "success": True,
                "summary": summary,
                "structure": structure
            }
        except Exception as e:
            logger.error(f"Error using AnythingLLM for summary: {str(e)}")
            # Fall back to OpenAI with chunking
            return self._generate_summary_with_openai_chunking(transcript, meeting_name)
    
    def _generate_summary_with_openai_chunking(self, transcript: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a meeting summary using OpenAI with chunking for long transcripts
        
        Args:
            transcript: The meeting transcript
            meeting_name: Optional name for the meeting
            
        Returns:
            Dictionary containing the summary and structured information
        """
        try:
            # Define max tokens for context window (conservative estimate)
            max_chunk_tokens = 3000  # Leave room for prompt and response
            
            # Estimate tokens in transcript (rough approximation: 4 chars ≈ 1 token)
            estimated_tokens = len(transcript) // 4
            
            if estimated_tokens > max_chunk_tokens:
                # Transcript is too long, need to chunk it
                logger.info(f"Transcript is approximately {estimated_tokens} tokens, chunking for processing")
                return self._process_long_transcript(transcript, meeting_name)
            else:
                # Transcript is short enough to process in one go
                return self._process_short_transcript(transcript, meeting_name)
        except Exception as e:
            logger.error(f"Error in OpenAI chunking: {str(e)}")
            return {
                "success": False,
                "error": f"Error processing transcript: {str(e)}"
            }
    
    def _process_short_transcript(self, transcript: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a transcript that fits within token limits"""
        # Create a system prompt for the meeting summary
        system_prompt = """
        You are a professional meeting summarizer. Your task is to create a comprehensive summary of the meeting transcript provided.
        
        Please structure your summary as follows:
        
        1. Meeting Overview: A brief 2-3 sentence overview of what the meeting was about.
        
        2. Key Points: Bullet points of the main topics discussed.
        
        3. Action Items: List any tasks, assignments, or follow-up actions mentioned in the meeting, including who is responsible and any deadlines.
        
        4. Decisions Made: List any decisions that were finalized during the meeting.
        
        5. Next Steps: What are the immediate next steps discussed in the meeting.
        
        Keep your summary concise but comprehensive, focusing on the most important information.
        """
        
        # Call the OpenAI API to generate the summary
        response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the meeting transcript:\n\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the summary from the response
        summary = response.choices[0].message.content
        
        # Now generate structured data for the summary
        structure_prompt = """
        Based on the meeting transcript, extract the following structured information in JSON format:
        
        {
            "meeting_name": "Auto-generated name if not provided",
            "key_points": [
                {"point": "Key point 1"},
                {"point": "Key point 2"}
            ],
            "action_items": [
                {"item": "Action item 1", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"},
                {"item": "Action item 2", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"}
            ],
            "decisions": [
                {"decision": "Decision 1"},
                {"decision": "Decision 2"}
            ],
            "next_steps": [
                {"step": "Next step 1"},
                {"step": "Next step 2"}
            ]
        }
        
        Only include the JSON in your response, nothing else.
        """
        
        # Call the OpenAI API to generate the structured data
        structure_response = openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": structure_prompt},
                {"role": "user", "content": f"Here is the meeting transcript:\n\n{transcript}"}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Extract the structured data from the response
        structure_text = structure_response.choices[0].message.content
        
        # Parse the JSON structure
        try:
            # Find JSON in the response
            import re
            json_match = re.search(r'({.*})', structure_text, re.DOTALL)
            if json_match:
                structure_text = json_match.group(1)
            
            structure = json.loads(structure_text)
            
            # Use provided meeting name if available
            if meeting_name:
                structure["meeting_name"] = meeting_name
        except Exception as e:
            logger.error(f"Error parsing structure JSON: {str(e)}")
            # Create a basic structure
            structure = {
                "meeting_name": meeting_name or "Meeting Summary",
                "key_points": [],
                "action_items": [],
                "decisions": [],
                "next_steps": []
            }
        
        return {
            "success": True,
            "summary": summary,
            "structure": structure
        }
    
    def _process_long_transcript(self, transcript: str, meeting_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a long transcript by chunking it into smaller pieces"""
        # Split the transcript into chunks
        chunks = self._split_into_chunks(transcript)
        logger.info(f"Split transcript into {len(chunks)} chunks for processing")
        
        # Process each chunk to get summaries
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create a system prompt for chunk summarization
            system_prompt = """
            You are a professional meeting summarizer. Your task is to extract the key information from this portion of a meeting transcript.
            
            Please extract:
            
            1. Key Points: What are the main topics discussed in this portion?
            
            2. Action Items: Any tasks, assignments, or follow-up actions mentioned, including who is responsible and deadlines.
            
            3. Decisions Made: Any decisions that were finalized during this portion.
            
            4. Important Context: Any critical context that would be needed to understand the rest of the meeting.
            
            Be concise but thorough. This is just one part of a longer transcript that will be combined later.
            """
            
            # Call the OpenAI API to summarize this chunk
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Here is a portion of the meeting transcript (part {i+1}/{len(chunks)}):\n\n{chunk}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                # Extract the summary from the response
                chunk_summary = response.choices[0].message.content
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                chunk_summaries.append(f"[Error processing this portion: {str(e)}]")
        
        # Combine the chunk summaries
        combined_summary = "\n\n".join([
            f"--- Part {i+1}/{len(chunks)} ---\n{summary}" 
            for i, summary in enumerate(chunk_summaries)
        ])
        
        # Generate a final summary from the combined chunk summaries
        final_system_prompt = """
        You are a professional meeting summarizer. Your task is to create a comprehensive summary of the meeting based on the provided chunk summaries.
        
        Please structure your summary as follows:
        
        1. Meeting Overview: A brief 2-3 sentence overview of what the meeting was about.
        
        2. Key Points: Bullet points of the main topics discussed.
        
        3. Action Items: List any tasks, assignments, or follow-up actions mentioned in the meeting, including who is responsible and any deadlines.
        
        4. Decisions Made: List any decisions that were finalized during the meeting.
        
        5. Next Steps: What are the immediate next steps discussed in the meeting.
        
        Keep your summary concise but comprehensive, focusing on the most important information.
        """
        
        # Call the OpenAI API to generate the final summary
        try:
            final_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": final_system_prompt},
                    {"role": "user", "content": f"Here are the summaries of different parts of a meeting transcript:\n\n{combined_summary}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract the summary from the response
            final_summary = final_response.choices[0].message.content
            
            # Now generate structured data for the summary
            structure_prompt = """
            Based on the meeting summaries, extract the following structured information in JSON format:
            
            {
                "meeting_name": "Auto-generated name if not provided",
                "key_points": [
                    {"point": "Key point 1"},
                    {"point": "Key point 2"}
                ],
                "action_items": [
                    {"item": "Action item 1", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"},
                    {"item": "Action item 2", "assignee": "Person name or 'Unassigned'", "deadline": "Deadline or 'None'"}
                ],
                "decisions": [
                    {"decision": "Decision 1"},
                    {"decision": "Decision 2"}
                ],
                "next_steps": [
                    {"step": "Next step 1"},
                    {"step": "Next step 2"}
                ]
            }
            
            Only include the JSON in your response, nothing else.
            """
            
            # Call the OpenAI API to generate the structured data
            structure_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": structure_prompt},
                    {"role": "user", "content": f"Here are the summaries of different parts of a meeting transcript:\n\n{combined_summary}"}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Extract the structured data from the response
            structure_text = structure_response.choices[0].message.content
            
            # Parse the JSON structure
            try:
                # Find JSON in the response
                import re
                json_match = re.search(r'({.*})', structure_text, re.DOTALL)
                if json_match:
                    structure_text = json_match.group(1)
                
                structure = json.loads(structure_text)
                
                # Use provided meeting name if available
                if meeting_name:
                    structure["meeting_name"] = meeting_name
            except Exception as e:
                logger.error(f"Error parsing structure JSON: {str(e)}")
                # Create a basic structure
                structure = {
                    "meeting_name": meeting_name or "Meeting Summary",
                    "key_points": [],
                    "action_items": [],
                    "decisions": [],
                    "next_steps": []
                }
            
            return {
                "success": True,
                "summary": final_summary,
                "structure": structure
            }
        except Exception as e:
            logger.error(f"Error generating final summary: {str(e)}")
            return {
                "success": False,
                "error": f"Error generating final summary: {str(e)}"
            }
    
    def _split_into_chunks(self, text: str, chunk_size: int = 3000) -> List[str]:
        """
        Split text into chunks of approximately equal size
        
        Args:
            text: Text to split
            chunk_size: Approximate size of each chunk in words
            
        Returns:
            List of text chunks
        """
        # Split text into words
        words = text.split()
        
        # Calculate number of chunks needed
        num_chunks = max(1, len(words) // chunk_size + (1 if len(words) % chunk_size > 0 else 0))
        
        # Calculate actual chunk size to make chunks more even
        actual_chunk_size = len(words) // num_chunks + (1 if len(words) % num_chunks > 0 else 0)
        
        # Create chunks
        chunks = []
        for i in range(0, len(words), actual_chunk_size):
            chunk = " ".join(words[i:i + actual_chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def format_meeting_minutes(self, summary_result: Dict[str, Any]) -> str:
        """
        Format the meeting minutes in a readable format
        
        Args:
            summary_result: The result from generate_summary
        
        Returns:
            Formatted meeting minutes as a string
        """
        if not summary_result.get("success"):
            return f"Error generating meeting minutes: {summary_result.get('error', 'Unknown error')}"
        
        summary = summary_result.get("summary", "")
        structure = summary_result.get("structure")
        
        if not structure:
            # If no structure is available, just return the summary
            return summary
        
        # Format the meeting minutes
        meeting_name = structure.get("meeting_name", "Meeting Minutes")
        
        formatted_minutes = f"# {meeting_name}\n\n"
        formatted_minutes += f"{summary}\n\n"
        
        # Add key points
        formatted_minutes += "## Key Points\n\n"
        for point in structure.get("key_points", []):
            formatted_minutes += f"- {point.get('point')}\n"
        
        formatted_minutes += "\n"
        
        # Add action items
        formatted_minutes += "## Action Items\n\n"
        for item in structure.get("action_items", []):
            assignee = item.get("assignee", "Unassigned")
            deadline = item.get("deadline", "None")
            formatted_minutes += f"- {item.get('item')} (Assignee: {assignee}, Deadline: {deadline})\n"
        
        formatted_minutes += "\n"
        
        # Add decisions
        formatted_minutes += "## Decisions Made\n\n"
        for decision in structure.get("decisions", []):
            formatted_minutes += f"- {decision.get('decision')}\n"
        
        formatted_minutes += "\n"
        
        # Add next steps
        formatted_minutes += "## Next Steps\n\n"
        for step in structure.get("next_steps", []):
            formatted_minutes += f"- {step.get('step')}\n"
        
        return formatted_minutes
