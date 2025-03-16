"""
Gradio Chatbot Interface with Meeting Minutes

This module provides a Gradio interface for the chatbot with meeting minutes functionality.
"""

import os
import tempfile
import logging
import json
import gradio as gr
from typing import Dict, Any, List, Optional, Tuple
import uuid
from datetime import datetime

from src.transcription.whisper_transcriber import OpenAIWhisperService, TranscriptionSettings
from src.meeting_minutes.meeting_summarizer import MeetingSummarizer
from src.meeting_minutes.lancedb_storage import LanceDBStorage
from src.anything_llm.anything_llm_client import AnythingLLMClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioChatbotWithMinutes:
    """
    Gradio interface for the meeting minutes generator with AnythingLLM chat integration
    """
    
    def __init__(self, port_range: Tuple[int, int] = (7680, 7690), transcript_storage=None):
        """
        Initialize the Gradio interface
        
        Args:
            port_range: Range of ports to try for the Gradio server
            transcript_storage: Storage implementation for transcripts and minutes
        """
        self.port_range = port_range
        
        # Initialize storage
        if transcript_storage is None:
            # Use default LanceDB storage if none provided
            self.transcript_storage = LanceDBStorage()
        else:
            # Use provided storage implementation
            self.transcript_storage = transcript_storage
        
        # Load settings
        self.settings = self.load_settings()
        
        # Initialize OpenAI client if API key is available
        self.openai_client = self.initialize_openai_client()
        
        # Initialize AnythingLLM client if settings are available
        self.anything_llm_client = self.initialize_anything_llm_client()
        
        # Current transcript data
        self.current_transcript = None
        self.current_transcript_info = None
        
        # Current minutes data
        self.current_minutes = None
        self.current_minutes_info = None
        
        # Create the interface
        self.interface = self.create_interface()
    
    def load_settings(self):
        """Load settings from file if it exists"""
        try:
            if os.path.exists("settings.json"):
                with open("settings.json", "r") as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Error loading settings: {str(e)}")
            return {}
    
    def initialize_openai_client(self):
        """Initialize OpenAI client if API key is available"""
        if self.settings.get("openai_api_key"):
            return OpenAIWhisperService(
                api_key=self.settings["openai_api_key"],
                model=self.settings.get("whisper_model", "whisper-1"),
                language=self.settings.get("language", "en")
            )
        return None
    
    def initialize_anything_llm_client(self):
        """Initialize AnythingLLM client if settings are available"""
        if self.settings.get("anything_llm_url") and self.settings.get("anything_llm_api_key"):
            return AnythingLLMClient(
                base_url=self.settings["anything_llm_url"],
                api_key=self.settings["anything_llm_api_key"],
                workspace_slug=self.settings.get("anything_llm_workspace", "default")
            )
        return None
    
    def save_settings(self, api_key, model, language, anything_llm_url, anything_llm_api_key, anything_llm_workspace):
        """
        Save settings to file
        
        Args:
            api_key: OpenAI API key
            model: Whisper model to use
            language: Language code
            anything_llm_url: AnythingLLM server URL
            anything_llm_api_key: AnythingLLM API key
            anything_llm_workspace: AnythingLLM workspace slug
        
        Returns:
            Success message
        """
        try:
            # Update settings
            self.settings = {
                "openai_api_key": api_key.strip() if api_key else "",
                "whisper_model": model.strip() if model else "whisper-1",
                "language": language.strip() if language else "en",
                "anything_llm_url": anything_llm_url.strip() if anything_llm_url else "http://localhost:3001",
                "anything_llm_api_key": anything_llm_api_key.strip() if anything_llm_api_key else "",
                "anything_llm_workspace": anything_llm_workspace.strip() if anything_llm_workspace else "default"
            }
            
            # Save to file
            try:
                with open("settings.json", "w") as f:
                    json.dump(self.settings, f, indent=2)
                logger.info("Settings saved successfully")
            except Exception as e:
                logger.error(f"Error saving settings to file: {str(e)}")
                return f"Error saving settings: {str(e)}"
            
            # Initialize OpenAI client
            try:
                if api_key:
                    self.openai_client = self.initialize_openai_client()
                    logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing OpenAI client: {str(e)}")
                return f"Error initializing OpenAI client: {str(e)}"
            
            # Initialize AnythingLLM client
            try:
                if anything_llm_url and anything_llm_api_key:
                    self.anything_llm_client = self.initialize_anything_llm_client()
                    logger.info("AnythingLLM client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing AnythingLLM client: {str(e)}")
                # Don't return error here, just log it - we can still function without AnythingLLM
            
            return "Settings saved successfully. You can now use the application."
        except Exception as e:
            logger.error(f"Error in save_settings: {str(e)}")
            return f"Error saving settings: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        interface = gr.Blocks(title="Meeting Minutes Generator", css="footer {visibility: hidden}")
        
        with interface:
            gr.Markdown("# Meeting Minutes Generator")
            
            with gr.Tabs():
                # Transcription tab
                with gr.TabItem("Transcription"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            audio_input = gr.Audio(
                                label="Record or upload audio",
                                sources=["microphone", "upload"],
                                type="filepath"
                            )
                            meeting_name_input = gr.Textbox(
                                label="Meeting Name (optional)",
                                placeholder="Enter a name for this meeting"
                            )
                            
                            with gr.Row():
                                transcribe_button = gr.Button("Transcribe")
                                clear_button = gr.Button("Clear")
                        
                        with gr.Column(scale=3):
                            transcript_output = gr.Textbox(
                                label="Transcript",
                                placeholder="Transcript will appear here",
                                lines=15,
                                max_lines=30
                            )
                            
                            transcript_info = gr.JSON(
                                label="Transcript Information",
                                visible=False
                            )
                            
                            with gr.Row():
                                summarize_button = gr.Button("Generate Meeting Minutes", interactive=False, variant="primary")
                    
                    gr.Markdown("### Recent Transcripts")
                    transcripts_list = gr.Dataframe(
                        headers=["ID", "Meeting Name", "Created At", "Duration (sec)"],
                        datatype=["str", "str", "str", "number"],
                        row_count=5,
                        col_count=(4, "fixed"),
                        interactive=False
                    )
                    
                    with gr.Row():
                        refresh_transcripts_button = gr.Button("Refresh")
                        load_transcript_button = gr.Button("Load Selected")
                        delete_transcript_button = gr.Button("Delete Selected")
                
                # Meeting Minutes tab
                with gr.TabItem("Meeting Minutes"):
                    gr.Markdown("## Meeting Minutes Generator")
                    gr.Markdown("Generate meeting minutes from transcripts.")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Add dropdown for selecting transcripts
                            transcript_dropdown = gr.Dropdown(
                                label="Select Transcript",
                                choices=self.get_transcript_choices(),
                                interactive=True
                            )
                            
                            # Refresh button for the dropdown
                            refresh_dropdown_button = gr.Button("Refresh Transcripts")
                            
                            minutes_transcript_id = gr.Textbox(
                                label="Transcript ID",
                                placeholder="ID of the transcript to summarize",
                                interactive=False,
                                visible=False  # Hide this since we're using the dropdown
                            )
                            minutes_meeting_name = gr.Textbox(
                                label="Meeting Name",
                                placeholder="Name of the meeting"
                            )
                            
                            generate_minutes_button = gr.Button("Generate Minutes")
                        
                        with gr.Column(scale=3):
                            minutes_transcript_text = gr.Textbox(
                                label="Transcript",
                                placeholder="Transcript text will appear here",
                                lines=10,
                                max_lines=15
                            )
                            
                            minutes_output = gr.Markdown(
                                label="Meeting Minutes",
                                value="Meeting minutes will appear here"
                            )
                            
                            minutes_info = gr.JSON(
                                label="Minutes Information",
                                visible=False
                            )
                    
                    gr.Markdown("### Recent Meeting Minutes")
                    minutes_list = gr.Dataframe(
                        headers=["ID", "Meeting Name", "Created At", "Transcript ID"],
                        datatype=["str", "str", "str", "str"],
                        row_count=5,
                        col_count=(4, "fixed"),
                        interactive=False
                    )
                    
                    with gr.Row():
                        refresh_minutes_button = gr.Button("Refresh")
                        load_minutes_button = gr.Button("Load Selected")
                        delete_minutes_button = gr.Button("Delete Selected")
                
                # Chat tab
                with gr.TabItem("Chat"):
                    self.create_chat_tab()
                
                # Settings tab
                with gr.TabItem("Settings"):
                    self.create_settings_tab()
            
            # Event handlers
            
            # Transcription tab
            transcribe_button.click(
                self.transcribe_audio,
                inputs=[audio_input, meeting_name_input],
                outputs=[transcript_output, transcript_info, summarize_button]
            )
            
            clear_button.click(
                self.clear_transcript,
                inputs=[],
                outputs=[transcript_output, transcript_info, summarize_button, audio_input]
            )
            
            summarize_button.click(
                self.summarize_transcript,
                inputs=[],
                outputs=[minutes_output, minutes_info, minutes_transcript_id, minutes_meeting_name, minutes_transcript_text]
            )
            
            refresh_transcripts_button.click(
                self.refresh_transcripts,
                inputs=[],
                outputs=[transcripts_list]
            )
            
            load_transcript_button.click(
                self.load_transcript,
                inputs=[transcripts_list],
                outputs=[transcript_output, transcript_info, summarize_button, minutes_transcript_id, minutes_meeting_name]
            )
            
            delete_transcript_button.click(
                self.delete_transcript,
                inputs=[transcripts_list],
                outputs=[transcripts_list]
            )
            
            # Meeting Minutes tab
            generate_minutes_button.click(
                self.generate_minutes,
                inputs=[transcript_dropdown, minutes_meeting_name],  # Use dropdown instead of transcript ID
                outputs=[minutes_output, minutes_info, minutes_transcript_text]
            )
            
            # Add event handler for transcript dropdown selection
            transcript_dropdown.change(
                self.load_transcript_for_minutes,
                inputs=[transcript_dropdown],
                outputs=[minutes_transcript_text, minutes_meeting_name]
            )
            
            # Add event handler for refreshing the transcript dropdown
            refresh_dropdown_button.click(
                self.refresh_transcript_dropdown,
                inputs=[],
                outputs=[transcript_dropdown]
            )
            
            refresh_minutes_button.click(
                self.refresh_minutes,
                inputs=[],
                outputs=[minutes_list]
            )
            
            load_minutes_button.click(
                self.load_minutes,
                inputs=[minutes_list],
                outputs=[minutes_output, minutes_info]
            )
            
            delete_minutes_button.click(
                self.delete_minutes,
                inputs=[minutes_list],
                outputs=[minutes_list]
            )
            
            # Initialize the interface
            interface.load(
                fn=self.refresh_transcripts,
                inputs=[],
                outputs=[transcripts_list]
            )
            
            interface.load(
                fn=self.refresh_minutes,
                inputs=[],
                outputs=[minutes_list]
            )
        
        return interface
    
    def create_settings_tab(self):
        """Create the settings tab"""
        with gr.Column():
            gr.Markdown("## Settings")
            
            with gr.Accordion("OpenAI Settings", open=True):
                api_key_input = gr.Textbox(
                    value=self.settings.get("openai_api_key", ""),
                    placeholder="Enter your OpenAI API key",
                    label="OpenAI API Key",
                    type="password"
                )
                
                whisper_model_input = gr.Dropdown(
                    choices=["whisper-1"],
                    value=self.settings.get("whisper_model", "whisper-1"),
                    label="Whisper Model"
                )
                
                language_input = gr.Textbox(
                    value=self.settings.get("language", "en"),
                    placeholder="Language code (e.g., en, fr, es)",
                    label="Language"
                )
            
            with gr.Accordion("AnythingLLM Settings", open=True):
                anything_llm_url_input = gr.Textbox(
                    value=self.settings.get("anything_llm_url", "http://localhost:3001"),
                    placeholder="AnythingLLM server URL",
                    label="AnythingLLM URL"
                )
                
                anything_llm_api_key_input = gr.Textbox(
                    value=self.settings.get("anything_llm_api_key", ""),
                    placeholder="AnythingLLM API key (if required)",
                    label="AnythingLLM API Key",
                    type="password"
                )
                
                anything_llm_workspace_input = gr.Textbox(
                    value=self.settings.get("anything_llm_workspace", "default"),
                    placeholder="AnythingLLM workspace slug",
                    label="AnythingLLM Workspace"
                )
                
                anything_llm_status = gr.Markdown("AnythingLLM Status: Not connected")
                
                def check_anything_llm_connection(url, api_key, workspace):
                    """Check connection to AnythingLLM"""
                    client = AnythingLLMClient(
                        base_url=url,
                        api_key=api_key,
                        workspace_slug=workspace
                    )
                    
                    if client.check_connection():
                        return "✅ AnythingLLM Status: Connected"
                    else:
                        return "❌ AnythingLLM Status: Not connected"
                
                check_connection_button = gr.Button("Check AnythingLLM Connection")
                check_connection_button.click(
                    check_anything_llm_connection,
                    inputs=[anything_llm_url_input, anything_llm_api_key_input, anything_llm_workspace_input],
                    outputs=[anything_llm_status]
                )
            
            save_button = gr.Button("Save Settings")
            save_status = gr.Markdown("")
            
            save_button.click(
                self.save_settings,
                inputs=[
                    api_key_input, 
                    whisper_model_input, 
                    language_input,
                    anything_llm_url_input,
                    anything_llm_api_key_input,
                    anything_llm_workspace_input
                ],
                outputs=[save_status]
            )
    
    def create_chat_tab(self):
        """Create the chat tab"""
        with gr.Column():
            gr.Markdown("## Chat with Meeting Information")
            gr.Markdown("Ask questions about your meeting transcripts and minutes using AnythingLLM.")
            
            # Connection status
            anything_llm_connection_status = gr.Markdown(
                "AnythingLLM Status: Checking connection..." if self.anything_llm_client else "AnythingLLM Status: Not configured"
            )
            
            # Check connection on tab load
            def update_connection_status():
                if self.anything_llm_client and self.check_anything_llm_connection():
                    return "✅ AnythingLLM Status: Connected"
                else:
                    return "❌ AnythingLLM Status: Not connected. Please check your settings."
            
            refresh_connection_button = gr.Button("Refresh Connection Status")
            refresh_connection_button.click(
                update_connection_status,
                inputs=[],
                outputs=[anything_llm_connection_status]
            )
            
            # Initialize connection status
            # anything_llm_connection_status.update(update_connection_status())
            
            # Chat interface
            chatbot = gr.Chatbot(height=400)
            
            with gr.Row():
                chat_input = gr.Textbox(
                    placeholder="Ask a question about your meetings...",
                    label="Question",
                    scale=9
                )
                chat_submit = gr.Button("Send", scale=1)
            
            with gr.Accordion("Chat Options", open=False):
                chat_mode = gr.Radio(
                    label="Chat Mode",
                    choices=["chat", "retrieval"],
                    value="retrieval",
                    info="Chat mode uses conversational context, Retrieval mode focuses on document search"
                )
                
                use_local_context = gr.Checkbox(
                    label="Include Local Meeting Data", 
                    value=True,
                    info="When enabled, includes data from local meeting storage in the prompt"
                )
            
            def chat_with_meetings(message, history, mode="chat", include_local_data=True):
                """Chat with meeting data"""
                if not message:
                    return history
                
                try:
                    # Check if we have an OpenAI API key
                    openai_api_key = self.settings.get("openai_api_key")
                    if not openai_api_key:
                        history.append((message, "OpenAI API key not set. Please add your API key in the Settings tab."))
                        return history
                    
                    # Prepare the message with local context if enabled
                    enhanced_message = message
                    system_message = "You are a helpful assistant that specializes in analyzing meeting transcripts and minutes."
                    local_context = ""
                    
                    # Check if the message is asking about a specific meeting by ID or name
                    specific_meeting_id = None
                    import re
                    
                    # Look for meeting ID patterns in the message
                    id_pattern = r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})'
                    id_matches = re.findall(id_pattern, message)
                    
                    # Look for meeting name patterns like "what happened in the demo"
                    name_pattern = r'what happened in(?: the)? (.+?)(?:\s*-|\s*\(|$)'
                    name_matches = re.search(name_pattern, message.lower())
                    
                    meeting_name_to_find = None
                    if name_matches:
                        meeting_name_to_find = name_matches.group(1).strip()
                    
                    # If we found potential meeting IDs or names, try to find the corresponding transcript
                    if id_matches or meeting_name_to_find:
                        logger.info(f"Looking for specific meeting: ID={id_matches}, Name={meeting_name_to_find}")
                        
                        # Get all transcripts
                        all_transcripts = self.transcript_storage.get_all_transcripts()
                        
                        if "transcripts" in all_transcripts:
                            for transcript in all_transcripts["transcripts"]:
                                # Check if this transcript matches any of our criteria
                                transcript_id = transcript.get("id", "")
                                
                                # Get the meeting name from the transcript
                                meeting_name = transcript.get("meeting_name", "")
                                if not meeting_name and isinstance(transcript.get("metadata"), dict):
                                    meeting_name = transcript["metadata"].get("meeting_name", "")
                                
                                # Check if this transcript matches our search criteria
                                id_match = any(tid in transcript_id for tid in id_matches)
                                name_match = meeting_name_to_find and meeting_name_to_find.lower() in meeting_name.lower()
                                
                                if id_match or name_match:
                                    logger.info(f"Found matching transcript: {transcript_id} - {meeting_name}")
                                    specific_meeting_id = transcript_id
                                    break
                    
                    # If we found a specific meeting, prioritize getting its content
                    if specific_meeting_id:
                        logger.info(f"Getting content for specific meeting: {specific_meeting_id}")
                        
                        # Get the transcript text
                        transcript_text = self.transcript_storage.get_transcript_text(specific_meeting_id)
                        
                        # Get the meeting name
                        transcript = self.transcript_storage.get_transcript(specific_meeting_id)
                        meeting_name = "Unnamed Meeting"
                        if transcript:
                            if "meeting_name" in transcript:
                                meeting_name = transcript.get("meeting_name")
                            elif "metadata" in transcript and isinstance(transcript["metadata"], dict):
                                meeting_name = transcript["metadata"].get("meeting_name", "Unnamed Meeting")
                        
                        # Get the meeting minutes if they exist
                        minutes_text = ""
                        minutes = self.transcript_storage.get_meeting_minutes_by_transcript_id(specific_meeting_id)
                        if minutes:
                            minutes_text = self.transcript_storage.get_meeting_minutes_text(minutes["id"])
                        
                        # Build the context with both transcript and minutes
                        if transcript_text or minutes_text:
                            context_parts = []
                            
                            if minutes_text:
                                # Truncate minutes text if it's too long
                                if len(minutes_text) > 2000:
                                    minutes_text = minutes_text[:2000] + "..."
                                context_parts.append(f"--- Meeting Minutes: {meeting_name} ---\n{minutes_text}\n")
                            
                            if transcript_text:
                                # Truncate transcript text if it's too long
                                if len(transcript_text) > 3000:
                                    transcript_text = transcript_text[:3000] + "..."
                                context_parts.append(f"--- Transcript: {meeting_name} ---\n{transcript_text}\n")
                            
                            local_context = "\n".join(context_parts)
                            logger.info(f"Added specific meeting context: {len(local_context)} characters")
                    
                    # If we don't have a specific meeting or if include_local_data is True, add general context
                    if not specific_meeting_id and include_local_data:
                        # Get local context if requested
                        if hasattr(self.transcript_storage, 'search_transcripts') and hasattr(self.transcript_storage, 'search_minutes'):
                            # Search transcripts
                            transcript_results = self.transcript_storage.search_transcripts(message, top_n=3, score_threshold=0.6)
                            
                            # Search minutes
                            minutes_results = self.transcript_storage.search_minutes(message, top_n=2, score_threshold=0.6)
                            
                            # Combine results
                            all_results = transcript_results + minutes_results
                            
                            # Sort by score
                            all_results.sort(key=lambda x: x["score"], reverse=True)
                            
                            # Build context string
                            context_parts = []
                            for result in all_results[:3]:  # Limit to top 3 results
                                result_type = "Transcript" if result["id"] in [r["id"] for r in transcript_results] else "Minutes"
                                meeting_name = result["metadata"].get("meeting_name", "Unnamed Meeting") if isinstance(result.get("metadata"), dict) else "Unnamed Meeting"
                                
                                # Truncate text to a reasonable length
                                text = result["text"]
                                if len(text) > 1000:
                                    text = text[:1000] + "..."
                                
                                context_parts.append(f"--- {result_type}: {meeting_name} ---\n{text}\n")
                            
                            local_context = "\n".join(context_parts)
                        else:
                            # Fallback to basic context retrieval
                            transcripts = self.transcript_storage.get_all_transcripts()
                            local_context = ""
                            
                            # First add meeting minutes for context
                            if hasattr(self.transcript_storage, 'get_all_meeting_minutes'):
                                minutes_data = self.transcript_storage.get_all_meeting_minutes()
                                if minutes_data and "minutes" in minutes_data:
                                    minutes_list = minutes_data["minutes"]
                                    # Sort by creation date if available
                                    if all(isinstance(m.get("created_at"), str) for m in minutes_list if "created_at" in m):
                                        try:
                                            minutes_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
                                        except Exception as e:
                                            logger.warning(f"Could not sort minutes by date: {str(e)}")
                                    
                                    # Add the most recent minutes first (up to 2)
                                    for minutes in minutes_list[:2]:
                                        minutes_id = minutes.get("id")
                                        if minutes_id:
                                            minutes_text = self.transcript_storage.get_meeting_minutes_text(minutes_id)
                                            if minutes_text:
                                                # Truncate text to a reasonable length
                                                if len(minutes_text) > 1000:
                                                    minutes_text = minutes_text[:1000] + "..."
                                                
                                                meeting_name = minutes.get("meeting_name", "")
                                                if not meeting_name and isinstance(minutes.get("metadata"), dict):
                                                    meeting_name = minutes["metadata"].get("meeting_name", "Unnamed Meeting")
                                                
                                                local_context += f"--- Meeting Minutes: {meeting_name} ---\n{minutes_text}\n\n"
                            
                            # Then add transcripts
                            if "transcripts" in transcripts:
                                for transcript in transcripts["transcripts"][:2]:  # Limit to 2 transcripts
                                    transcript_id = transcript.get("id")
                                    if transcript_id:
                                        transcript_text = self.transcript_storage.get_transcript_text(transcript_id)
                                        if transcript_text:
                                            # Truncate text to a reasonable length
                                            if len(transcript_text) > 1000:
                                                transcript_text = transcript_text[:1000] + "..."
                                            
                                            meeting_name = transcript.get("meeting_name", "")
                                            if not meeting_name and isinstance(transcript.get("metadata"), dict):
                                                meeting_name = transcript["metadata"].get("meeting_name", "Unnamed Meeting")
                                            
                                            local_context += f"--- Transcript: {meeting_name} ---\n{transcript_text}\n\n"
                    
                    if local_context:
                        system_message = f"""You are a helpful assistant that specializes in analyzing meeting transcripts and minutes.
                        
                        Here is some relevant context from meeting transcripts and minutes:
                        
                        {local_context}
                        
                        Use this context to answer the user's questions when relevant."""
                        enhanced_message = f"Based on the provided context, please answer: {message}"
                        logger.info(f"Added context to prompt: {len(local_context)} characters")
                    
                    # First try AnythingLLM if it's available
                    anything_llm_working = False
                    if self.anything_llm_client and self.check_anything_llm_connection():
                        try:
                            # Prepare message for AnythingLLM
                            if local_context:
                                anything_llm_message = f"I'm providing some relevant meeting information for context:\n\n{local_context}\n\nBased on this context, please answer: {message}"
                            else:
                                anything_llm_message = message
                            
                            # Send chat request to AnythingLLM
                            response_data = self.anything_llm_client.chat(anything_llm_message, mode=mode)
                            
                            if response_data.get("success"):
                                response = response_data.get("response", "")
                                if response.strip():  # Check if we got a non-empty response
                                    anything_llm_working = True
                                    return history + [(message, response)]
                        except Exception as e:
                            logger.warning(f"AnythingLLM chat failed, falling back to OpenAI: {str(e)}")
                    
                    # If AnythingLLM failed or is not available, use OpenAI directly
                    if not anything_llm_working:
                        logger.info("Using OpenAI for chat (AnythingLLM not available or failed)")
                        
                        # Create messages for OpenAI chat
                        messages = [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": enhanced_message}
                        ]
                        
                        # Add chat history for context if in chat mode
                        if mode == "chat" and len(history) > 1:
                            # Add up to 5 previous exchanges (10 messages) for context
                            for i in range(max(0, len(history) - 5), len(history)):
                                if i < len(history):
                                    user_msg, assistant_msg = history[i]
                                    if user_msg:  # User message
                                        messages.append({"role": "user", "content": user_msg})
                                    if assistant_msg:  # Assistant message
                                        messages.append({"role": "assistant", "content": assistant_msg})
                        
                        try:
                            # Send chat request to OpenAI
                            import openai
                            client = openai.OpenAI(api_key=openai_api_key)
                            
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",  # You can change this to a different model if needed
                                messages=messages,
                                temperature=0.7,
                                max_tokens=1000
                            )
                            
                            response_text = response.choices[0].message.content
                            return history + [(message, response_text)]
                        except Exception as e:
                            logger.error(f"Error with OpenAI chat: {str(e)}")
                            return history + [(message, f"Error with OpenAI chat: {str(e)}")]
                    
                    return history
                except Exception as e:
                    logger.error(f"Error in chat: {str(e)}")
                    return history + [(message, f"Error: {str(e)}")]
            
            # Connect the chat input and submit button
            chat_input.submit(
                chat_with_meetings,
                inputs=[chat_input, chatbot, chat_mode, use_local_context],
                outputs=[chatbot]
            )
            
            chat_submit.click(
                chat_with_meetings,
                inputs=[chat_input, chatbot, chat_mode, use_local_context],
                outputs=[chatbot]
            )
            
            # Clear chat button
            clear_chat = gr.Button("Clear Chat")
            clear_chat.click(lambda: None, None, chatbot, queue=False)
            
            # Upload transcript to AnythingLLM section
            with gr.Accordion("Upload to AnythingLLM", open=False):
                gr.Markdown("Upload meeting transcripts and minutes to AnythingLLM for improved retrieval.")
                
                transcript_dropdown = gr.Dropdown(
                    label="Select Transcript",
                    choices=self.get_transcript_choices(),
                    interactive=True
                )
                
                upload_button = gr.Button("Upload to AnythingLLM")
                upload_status = gr.Markdown("")
                
                def upload_to_anything_llm(transcript_id):
                    """Upload a transcript and its minutes to AnythingLLM or save to OpenAI as fallback"""
                    if not transcript_id:
                        return "Please select a transcript to upload."
                    
                    # Check if we have an OpenAI API key
                    openai_api_key = self.settings.get("openai_api_key")
                    if not openai_api_key:
                        return "OpenAI API key not set. Please add your API key in the Settings tab."
                    
                    try:
                        # Get the transcript and minutes
                        transcript = self.transcript_storage.get_transcript(transcript_id)
                        if not transcript:
                            return "Transcript not found."
                        
                        transcript_text = self.transcript_storage.get_transcript_text(transcript_id)
                        if not transcript_text:
                            return "Transcript text not found."
                        
                        # Check if there are meeting minutes
                        minutes = self.transcript_storage.get_meeting_minutes_by_transcript_id(transcript_id)
                        minutes_text = ""
                        if minutes:
                            minutes_text = self.transcript_storage.get_meeting_minutes_text(minutes["id"])
                        
                        # Get meeting name from transcript metadata
                        meeting_name = "Unnamed Meeting"
                        if "meeting_name" in transcript:
                            meeting_name = transcript.get("meeting_name")
                        elif "metadata" in transcript and isinstance(transcript["metadata"], dict):
                            meeting_name = transcript["metadata"].get("meeting_name", "Unnamed Meeting")
                        
                        # Create a temporary file with the content
                        temp_file_path = f"temp_meeting_{transcript_id}.md"
                        
                        with open(temp_file_path, "w", encoding="utf-8") as f:
                            f.write(f"# {meeting_name}\n\n")
                            
                            if minutes_text:
                                f.write(f"## Meeting Minutes\n\n{minutes_text}\n\n")
                            
                            f.write(f"## Full Transcript\n\n{transcript_text}\n\n")
                        
                        # Try to upload to AnythingLLM first if it's connected
                        anything_llm_success = False
                        anything_llm_error = ""
                        
                        if self.anything_llm_client and self.check_anything_llm_connection():
                            logger.info(f"Uploading {meeting_name} to AnythingLLM...")
                            result = self.anything_llm_client.upload_document(
                                file_path=temp_file_path,
                                file_name=f"{meeting_name}.md"
                            )
                            
                            if result and result.get("success", False):
                                anything_llm_success = True
                            else:
                                error = result.get("error", "Unknown error")
                                anything_llm_error = error
                                logger.error(f"Failed to upload to AnythingLLM: {error}")
                        else:
                            anything_llm_error = "AnythingLLM is not connected"
                            logger.warning("AnythingLLM is not connected, using OpenAI API as fallback")
                        
                        # If AnythingLLM upload failed or not connected, use OpenAI API
                        openai_success = False
                        openai_error = ""
                        
                        if not anything_llm_success:
                            try:
                                # Use OpenAI API to process the content
                                import openai
                                client = openai.OpenAI(api_key=openai_api_key)
                                
                                # Read the content from the temporary file
                                with open(temp_file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                
                                # Create a file in OpenAI for reference
                                file_response = client.files.create(
                                    file=open(temp_file_path, "rb"),
                                    purpose="assistants"
                                )
                                
                                file_id = file_response.id
                                logger.info(f"File uploaded to OpenAI with ID: {file_id}")
                                
                                # Store the file ID in the transcript metadata for future reference
                                if "metadata" not in transcript:
                                    transcript["metadata"] = {}
                                
                                if not isinstance(transcript["metadata"], dict):
                                    transcript["metadata"] = {}
                                
                                transcript["metadata"]["openai_file_id"] = file_id
                                self.transcript_storage.update_transcript(transcript_id, transcript)
                                
                                openai_success = True
                            except Exception as e:
                                openai_error = str(e)
                                logger.error(f"Error using OpenAI API as fallback: {str(e)}")
                        
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        
                        # Return appropriate status message
                        if anything_llm_success and openai_success:
                            return f"✅ Successfully uploaded {meeting_name} to both AnythingLLM and OpenAI."
                        elif anything_llm_success:
                            return f"✅ Successfully uploaded {meeting_name} to AnythingLLM."
                        elif openai_success:
                            return f"✅ Successfully saved {meeting_name} to OpenAI. AnythingLLM upload failed: {anything_llm_error}"
                        else:
                            return f"❌ Failed to upload to AnythingLLM: {anything_llm_error}. OpenAI fallback also failed: {openai_error}"
                    
                    except Exception as e:
                        logger.error(f"Error in upload process: {str(e)}")
                        return f"❌ Error: {str(e)}"
                
                upload_button.click(
                    upload_to_anything_llm,
                    inputs=[transcript_dropdown],
                    outputs=[upload_status]
                )
    
    def get_transcript_choices(self):
        """Get a list of transcript choices for the dropdown"""
        try:
            result = self.transcript_storage.get_all_transcripts()
            if not result or not isinstance(result, dict) or "transcripts" not in result:
                logger.warning("No transcripts found or invalid result format")
                return [("Select a transcript", "")]
            
            transcripts = result.get("transcripts", [])
            
            choices = [("Select a transcript", "")]  # Add a default option
            for transcript in transcripts:
                # Get meeting name from metadata if available
                meeting_name = "Unnamed Meeting"
                if "meeting_name" in transcript:
                    meeting_name = transcript.get("meeting_name")
                elif "metadata" in transcript and isinstance(transcript["metadata"], dict):
                    meeting_name = transcript["metadata"].get("meeting_name", "Unnamed Meeting")
                
                # Get creation date if available
                created_at = ""
                if "created_at" in transcript:
                    created_at = transcript.get("created_at")
                elif "metadata" in transcript and isinstance(transcript["metadata"], dict):
                    created_at = transcript["metadata"].get("created_at", "")
                
                # Format the date if it exists
                date_str = ""
                if created_at:
                    try:
                        # Try to parse the date and format it
                        if isinstance(created_at, str):
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            date_str = f" - {dt.strftime('%Y-%m-%d %H:%M')}"
                    except Exception as e:
                        logger.warning(f"Could not parse date: {str(e)}")
                
                transcript_id = transcript.get("id")
                if transcript_id:
                    choices.append((f"{meeting_name}{date_str} ({transcript_id})", transcript_id))
            
            return choices
        except Exception as e:
            logger.error(f"Error getting transcript choices: {str(e)}")
            return [("Select a transcript", "")]
    
    def transcribe_audio(self, audio_path: str, meeting_name: Optional[str] = None) -> Tuple[str, Dict[str, Any], bool]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            meeting_name: Optional name for the meeting
        
        Returns:
            Tuple of (transcript text, transcript info, summarize button enabled)
        """
        if not audio_path:
            return "No audio file provided", None, False
        
        if not self.openai_client:
            return "OpenAI API key not set. Please configure it in the Settings tab.", None, False
        
        try:
            # Transcribe and store the audio
            result = self.openai_client.transcribe_and_store(
                audio_file_path=audio_path,
                transcript_storage=self.transcript_storage,
                meeting_name=meeting_name
            )
            
            # Debug log to check the result
            logger.info(f"Transcription result: {result}")
            
            if not result.get("success"):
                error_message = result.get("error", "Unknown error")
                logger.error(f"Transcription failed: {error_message}")
                return f"Transcription failed: {error_message}", None, False
            
            # Store the current transcript ID and meeting name
            self.current_transcript_id = result.get("transcript_id")
            self.current_meeting_name = result.get("meeting_name")
            
            # Debug log to check if button should be enabled
            logger.info(f"Transcription successful, enabling summarize button. ID: {self.current_transcript_id}")
            
            # Return the transcript text and info
            return result.get("text"), {
                "transcript_id": self.current_transcript_id,
                "meeting_name": self.current_meeting_name,
                "duration": result.get("duration"),
                "model": result.get("model"),
                "language": result.get("language")
            }, True
        
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return f"Transcription failed: {str(e)}", None, False
    
    def clear_transcript(self) -> Tuple[str, None, bool, None]:
        """
        Clear the transcript
        
        Returns:
            Tuple of (empty transcript, None info, disabled summarize button, None audio)
        """
        self.current_transcript_id = None
        self.current_meeting_name = None
        return "", None, False, None
    
    def summarize_transcript(self) -> Tuple[str, Dict[str, Any], str, str, str]:
        """
        Summarize the current transcript
        
        Returns:
            Tuple of (meeting minutes, minutes info, transcript ID, meeting name, transcript text)
        """
        logger.info(f"Attempting to summarize transcript with ID: {self.current_transcript_id}")
        
        if not self.current_transcript_id:
            logger.warning("No transcript ID available for summarization")
            return "No transcript to summarize", None, "", "", ""
        
        if not self.openai_client:
            logger.warning("OpenAI client not initialized")
            return "OpenAI API key not set. Please configure it in the Settings tab.", None, "", "", ""
        
        try:
            # Get the transcript text
            transcript_text = self.transcript_storage.get_transcript_text(self.current_transcript_id)
            logger.info(f"Retrieved transcript text of length: {len(transcript_text) if transcript_text else 0}")
            
            # Generate summary from the transcript ID
            summarizer = MeetingSummarizer()
            result = summarizer.generate_summary_from_transcript_id(
                transcript_id=self.current_transcript_id,
                transcript_storage=self.transcript_storage,
                meeting_name=self.current_meeting_name
            )
            
            logger.info(f"Summarization result: {result}")
            
            if not result.get("success"):
                error_message = result.get("error", "Unknown error")
                logger.error(f"Summarization failed: {error_message}")
                return f"Summarization failed: {error_message}", None, "", "", ""
            
            # Return the formatted minutes and info
            formatted_minutes = result.get("formatted_minutes", "")
            minutes_info = {
                "minutes_id": result.get("minutes_id", ""),
                "transcript_id": result.get("transcript_id", ""),
                "meeting_name": result.get("meeting_name", ""),
                "structure": result.get("structure", {})
            }
            
            logger.info(f"Successfully generated minutes with ID: {minutes_info['minutes_id']}")
            
            return formatted_minutes, minutes_info, self.current_transcript_id, self.current_meeting_name, transcript_text
        
        except Exception as e:
            logger.error(f"Summarization failed with exception: {str(e)}")
            return f"Summarization failed: {str(e)}", None, "", "", ""
    
    def refresh_transcripts(self) -> List[List[Any]]:
        """
        Refresh the list of transcripts
        
        Returns:
            List of transcripts
        """
        try:
            logger.info("Refreshing transcripts list...")
            result = self.transcript_storage.get_all_transcripts()
            
            # Debug log the result
            logger.info(f"Transcript result: {result}")
            
            # The result structure is different in LanceDB implementation
            transcripts = result.get("transcripts", [])
            
            # Format the transcripts for the dataframe
            formatted_transcripts = []
            for transcript in transcripts:
                logger.info(f"Processing transcript: {transcript}")
                formatted_transcripts.append([
                    transcript.get("id", ""),
                    transcript.get("meeting_name", "Unnamed Meeting"),
                    transcript.get("created_at", ""),
                    transcript.get("duration", 0)
                ])
            
            logger.info(f"Formatted transcripts: {formatted_transcripts}")
            return formatted_transcripts
        
        except Exception as e:
            logger.error(f"Failed to refresh transcripts: {str(e)}")
            # Return a single row with error information to make it visible in the UI
            return [["Error", f"Failed to load transcripts: {str(e)}", "", 0]]
    
    def load_transcript(self, selected_row: Any) -> Tuple[str, Dict[str, Any], bool, str, str]:
        """
        Load a transcript from the selected row
        
        Args:
            selected_row: Selected row from the dataframe
        
        Returns:
            Tuple of (transcript text, transcript info, summarize button enabled, transcript ID, meeting name)
        """
        logger.info(f"Loading transcript, selected_row type: {type(selected_row)}")
        
        # Handle DataFrame input (when user selects a row in the UI)
        if hasattr(selected_row, 'iloc') and hasattr(selected_row, 'empty'):  # It's a DataFrame
            if selected_row.empty:
                return "No transcript selected", None, False, "", ""
            # Get the first row, first column (transcript ID)
            try:
                transcript_id = selected_row.iloc[0, 0]
                logger.info(f"Selected transcript ID from DataFrame: {transcript_id}")
            except Exception as e:
                logger.error(f"Error extracting transcript ID from DataFrame: {str(e)}")
                return f"Failed to extract transcript ID: {str(e)}", None, False, "", ""
        elif isinstance(selected_row, list) and len(selected_row) > 0:
            # Handle list input (direct API call)
            transcript_id = selected_row[0]
            logger.info(f"Selected transcript ID from list: {transcript_id}")
        else:
            logger.warning(f"Invalid selected_row format: {type(selected_row)}")
            return "No transcript selected", None, False, "", ""
        
        try:
            result = self.transcript_storage.get_transcript(transcript_id)
            
            if not result.get("success"):
                error_message = result.get("error", "Unknown error")
                return f"Failed to load transcript: {error_message}", None, False, "", ""
            
            transcript = result.get("transcript")
            
            # Store the current transcript ID and meeting name
            self.current_transcript_id = transcript.get("id")
            self.current_meeting_name = transcript.get("meeting_name")
            
            # Return the transcript text and info
            return transcript.get("text"), {
                "transcript_id": self.current_transcript_id,
                "meeting_name": self.current_meeting_name,
                "duration": transcript.get("duration"),
                "source": transcript.get("source"),
                "created_at": transcript.get("created_at"),
                "metadata": transcript.get("metadata")
            }, True, self.current_transcript_id, self.current_meeting_name
        
        except Exception as e:
            logger.error(f"Failed to load transcript: {str(e)}")
            return f"Failed to load transcript: {str(e)}", None, False, "", ""
    
    def delete_transcript(self, selected_row: Any) -> List[List[Any]]:
        """
        Delete a transcript from the selected row
        
        Args:
            selected_row: Selected row from the dataframe
        
        Returns:
            Updated list of transcripts
        """
        logger.info(f"Deleting transcript, selected_row type: {type(selected_row)}")
        
        # Handle DataFrame input (when user selects a row in the UI)
        if hasattr(selected_row, 'iloc') and hasattr(selected_row, 'empty'):  # It's a DataFrame
            if selected_row.empty:
                logger.warning("No row selected for deletion")
                return self.refresh_transcripts()
            # Get the first row, first column (transcript ID)
            try:
                transcript_id = selected_row.iloc[0, 0]
                logger.info(f"Selected transcript ID for deletion from DataFrame: {transcript_id}")
            except Exception as e:
                logger.error(f"Error extracting transcript ID for deletion from DataFrame: {str(e)}")
                return self.refresh_transcripts()
        elif isinstance(selected_row, list) and len(selected_row) > 0:
            # Handle list input (direct API call)
            transcript_id = selected_row[0]
            logger.info(f"Selected transcript ID for deletion from list: {transcript_id}")
        else:
            logger.warning(f"Invalid selected_row format for deletion: {type(selected_row)}")
            return self.refresh_transcripts()
        
        try:
            # The delete_transcript method returns a boolean, not a dictionary
            success = self.transcript_storage.delete_transcript(transcript_id)
            
            if not success:
                logger.error(f"Failed to delete transcript: {transcript_id}")
            else:
                logger.info(f"Successfully deleted transcript: {transcript_id}")
                # Clear current transcript if it was the one deleted
                if self.current_transcript_id == transcript_id:
                    self.current_transcript_id = None
                    self.current_meeting_name = None
            
            # Refresh the list of transcripts
            return self.refresh_transcripts()
        
        except Exception as e:
            logger.error(f"Failed to delete transcript: {str(e)}")
            return self.refresh_transcripts()
    
    def generate_minutes(self, transcript_id: str, meeting_name: Optional[str] = None) -> Tuple[str, Dict[str, Any], str]:
        """
        Generate meeting minutes for a transcript
        
        Args:
            transcript_id: ID of the transcript
            meeting_name: Optional name for the meeting
        
        Returns:
            Tuple of (meeting minutes, minutes info, transcript text)
        """
        logger.info(f"Generating minutes for transcript ID: {transcript_id}")
        
        if not transcript_id:
            logger.warning("No transcript ID provided for generating minutes")
            return "Please select a transcript from the dropdown", None, ""
        
        try:
            # Get the transcript text
            transcript_data = self.transcript_storage.get_transcript(transcript_id)
            if not transcript_data:
                logger.error(f"Failed to retrieve transcript with ID: {transcript_id}")
                return f"Failed to load transcript: Transcript with ID {transcript_id} not found", None, ""
            
            transcript_text = transcript_data.get("text", "")
            if not transcript_text:
                logger.error(f"No text found in transcript with ID: {transcript_id}")
                return "Failed to load transcript: No text found in transcript", None, ""
            
            # Use the provided meeting name or get it from the transcript
            if not meeting_name:
                meeting_name = transcript_data.get("meeting_name", "")
            
            # Generate summary
            summarizer = MeetingSummarizer()
            
            # Check if OpenAI API key is set
            if not summarizer.is_available():
                return "OpenAI API key not set. Please configure it in the Settings tab.", None, transcript_text
            
            result = summarizer.generate_summary_from_transcript_id(
                transcript_id=transcript_id,
                transcript_storage=self.transcript_storage,
                meeting_name=meeting_name
            )
            
            if not result.get("success"):
                error_message = result.get("error", "Unknown error")
                logger.error(f"Failed to generate minutes: {error_message}")
                return f"Failed to generate minutes: {error_message}", None, transcript_text
            
            # Return the formatted minutes and info
            formatted_minutes = result.get("formatted_minutes", "")
            minutes_info = {
                "minutes_id": result.get("minutes_id", ""),
                "transcript_id": result.get("transcript_id", ""),
                "meeting_name": result.get("meeting_name", ""),
                "structure": result.get("structure", {})
            }
            
            return formatted_minutes, minutes_info, transcript_text
        
        except Exception as e:
            logger.error(f"Error generating minutes: {str(e)}")
            return f"Error generating minutes: {str(e)}", None, ""
    
    def refresh_minutes(self) -> List[List[Any]]:
        """
        Refresh the list of meeting minutes
        
        Returns:
            List of meeting minutes
        """
        try:
            # Use get_all_minutes instead of get_all_meeting_minutes
            result = self.transcript_storage.get_all_minutes()
            
            # The result structure is different in LanceDB implementation
            minutes = result.get("minutes", [])
            
            # Format the minutes for the dataframe
            formatted_minutes = []
            for minute in minutes:
                formatted_minutes.append([
                    minute.get("id"),
                    minute.get("meeting_name"),
                    minute.get("created_at"),
                    minute.get("transcript_id")
                ])
            
            return formatted_minutes
        
        except Exception as e:
            logger.error(f"Failed to refresh meeting minutes: {str(e)}")
            return []
    
    def load_minutes(self, selected_row: Any) -> Tuple[str, Dict[str, Any]]:
        """
        Load meeting minutes from the selected row
        
        Args:
            selected_row: Selected row from the dataframe
        
        Returns:
            Tuple of (minutes text, minutes info)
        """
        logger.info(f"Loading minutes, selected_row type: {type(selected_row)}")
        
        # Handle DataFrame input (when user selects a row in the UI)
        if hasattr(selected_row, 'iloc') and hasattr(selected_row, 'empty'):  # It's a DataFrame
            if selected_row.empty:
                return "No minutes selected", None
            # Get the first row, first column (minutes ID)
            try:
                minutes_id = selected_row.iloc[0, 0]
                logger.info(f"Selected minutes ID from DataFrame: {minutes_id}")
            except Exception as e:
                logger.error(f"Error extracting minutes ID from DataFrame: {str(e)}")
                return f"Failed to extract minutes ID: {str(e)}", None
        elif isinstance(selected_row, list) and len(selected_row) > 0:
            # Handle list input (direct API call)
            minutes_id = selected_row[0]
            logger.info(f"Selected minutes ID from list: {minutes_id}")
        else:
            logger.warning(f"Invalid selected_row format: {type(selected_row)}")
            return "No minutes selected", None
        
        try:
            result = self.transcript_storage.get_minutes(minutes_id)
            
            if not result:
                return "Failed to load minutes", None
            
            # Return the minutes text and info
            return result.get("content"), {
                "minutes_id": result.get("id"),
                "meeting_name": result.get("meeting_name"),
                "transcript_id": result.get("transcript_id"),
                "created_at": result.get("created_at"),
                "metadata": result.get("metadata")
            }
        
        except Exception as e:
            logger.error(f"Failed to load minutes: {str(e)}")
            return f"Failed to load minutes: {str(e)}", None
    
    def delete_minutes(self, selected_row: Any) -> List[List[Any]]:
        """
        Delete meeting minutes from the selected row
        
        Args:
            selected_row: Selected row from the dataframe
        
        Returns:
            Updated list of minutes
        """
        logger.info(f"Deleting minutes, selected_row type: {type(selected_row)}")
        
        # Handle DataFrame input (when user selects a row in the UI)
        if hasattr(selected_row, 'iloc') and hasattr(selected_row, 'empty'):  # It's a DataFrame
            if selected_row.empty:
                logger.warning("No row selected for minutes deletion")
                return self.refresh_minutes()
            # Get the first row, first column (minutes ID)
            try:
                minutes_id = selected_row.iloc[0, 0]
                logger.info(f"Selected minutes ID for deletion from DataFrame: {minutes_id}")
            except Exception as e:
                logger.error(f"Error extracting minutes ID for deletion from DataFrame: {str(e)}")
                return self.refresh_minutes()
        elif isinstance(selected_row, list) and len(selected_row) > 0:
            # Handle list input (direct API call)
            minutes_id = selected_row[0]
            logger.info(f"Selected minutes ID for deletion from list: {minutes_id}")
        else:
            logger.warning(f"Invalid selected_row format for minutes deletion: {type(selected_row)}")
            return self.refresh_minutes()
        
        try:
            success = self.transcript_storage.delete_minutes(minutes_id)
            
            if not success:
                logger.error(f"Failed to delete minutes: {minutes_id}")
            else:
                logger.info(f"Successfully deleted minutes: {minutes_id}")
            
            # Refresh the list of minutes
            return self.refresh_minutes()
        
        except Exception as e:
            logger.error(f"Failed to delete minutes: {str(e)}")
            return self.refresh_minutes()
    
    def refresh_transcript_dropdown(self) -> List[Tuple[str, str]]:
        """
        Refresh the transcript dropdown
        
        Returns:
            List of transcript choices
        """
        return self.get_transcript_choices()
    
    def load_transcript_for_minutes(self, transcript_id: str) -> Tuple[str, str]:
        """
        Load a transcript for the meeting minutes page
        
        Args:
            transcript_id: ID of the transcript
        
        Returns:
            Tuple of (transcript text, meeting name)
        """
        logger.info(f"Loading transcript for minutes with ID: {transcript_id}")
        
        if not transcript_id:
            logger.warning("No transcript ID provided")
            return "No transcript selected", ""
        
        try:
            # First try to get the transcript text directly
            transcript_text = self.transcript_storage.get_transcript_text(transcript_id)
            
            if not transcript_text:
                logger.warning(f"No text found using get_transcript_text for ID: {transcript_id}")
                # Try getting the full transcript object as fallback
                transcript_data = self.transcript_storage.get_transcript(transcript_id)
                
                if not transcript_data:
                    logger.error(f"Failed to retrieve transcript with ID: {transcript_id}")
                    return f"Failed to load transcript: Transcript with ID {transcript_id} not found", ""
                
                # Extract text from the transcript data
                transcript_text = transcript_data.get("text", "")
                if not transcript_text:
                    logger.error(f"No text found in transcript data for ID: {transcript_id}")
                    return "Failed to load transcript: No text found in transcript", ""
            
            # Get the meeting name
            meeting_name = ""
            try:
                # Try to get the meeting name from the transcript data
                transcript_data = self.transcript_storage.get_transcript(transcript_id)
                if transcript_data:
                    # Check if meeting_name is in the root or in metadata
                    if "meeting_name" in transcript_data:
                        meeting_name = transcript_data.get("meeting_name")
                    elif "metadata" in transcript_data and isinstance(transcript_data["metadata"], dict):
                        meeting_name = transcript_data["metadata"].get("meeting_name", "")
            except Exception as e:
                logger.warning(f"Could not get meeting name for transcript {transcript_id}: {str(e)}")
            
            logger.info(f"Successfully loaded transcript for minutes with ID: {transcript_id}")
            return transcript_text, meeting_name
        
        except Exception as e:
            logger.error(f"Error loading transcript for minutes: {str(e)}")
            return f"Failed to load transcript: {str(e)}", ""
    
    def check_anything_llm_connection(self):
        """
        Check if AnythingLLM is connected and working
        
        Returns:
            bool: True if connected, False otherwise
        """
        if not self.anything_llm_client:
            return False
        
        try:
            # Try to ping the AnythingLLM server
            result = self.anything_llm_client.check_connection()
            if result:
                logger.info("AnythingLLM connection successful")
                return True
            else:
                logger.warning("AnythingLLM connection check failed")
                return False
        except Exception as e:
            logger.warning(f"Error checking AnythingLLM connection: {str(e)}")
            return False
    
    def launch(self, inbrowser=True):
        """
        Launch the Gradio interface
        
        Args:
            inbrowser: Whether to open the interface in a browser
        """
        try:
            # Let Gradio find an available port within the range
            self.interface.launch(
                server_port=self.port_range[0],
                server_name="0.0.0.0",  # Allow external connections
                inbrowser=inbrowser,
                share=False,
                quiet=True,
                prevent_thread_lock=False
            )
        except Exception as e:
            logger.error(f"Failed to launch on initial port {self.port_range[0]}: {str(e)}")
            # Try other ports in the range
            for port in range(self.port_range[0] + 1, self.port_range[1] + 1):
                try:
                    logger.info(f"Trying port {port}...")
                    self.interface.launch(
                        server_port=port,
                        server_name="0.0.0.0",  # Allow external connections
                        inbrowser=inbrowser,
                        share=False,
                        quiet=True,
                        prevent_thread_lock=False
                    )
                    return
                except Exception as e:
                    logger.error(f"Failed to launch on port {port}: {str(e)}")
            
            logger.error("Failed to launch on any port in the range")
            raise RuntimeError(f"Could not find an available port in range {self.port_range[0]}-{self.port_range[1]}")

def main():
    """Main entry point for the application"""
    app = GradioChatbotWithMinutes()
    app.launch()

if __name__ == "__main__":
    main()
