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
                "openai_api_key": api_key,
                "whisper_model": model,
                "language": language,
                "anything_llm_url": anything_llm_url,
                "anything_llm_api_key": anything_llm_api_key,
                "anything_llm_workspace": anything_llm_workspace
            }
            
            # Save to file
            with open("settings.json", "w") as f:
                json.dump(self.settings, f)
            
            # Initialize OpenAI client
            self.openai_client = self.initialize_openai_client()
            
            # Initialize AnythingLLM client
            self.anything_llm_client = self.initialize_anything_llm_client()
            
            return "Settings saved successfully"
        except Exception as e:
            logger.error(f"Error saving settings: {str(e)}")
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
                    with gr.Row():
                        with gr.Column(scale=1):
                            minutes_transcript_id = gr.Textbox(
                                label="Transcript ID",
                                placeholder="ID of the transcript to summarize",
                                interactive=False
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
                inputs=[minutes_transcript_id, minutes_meeting_name],
                outputs=[minutes_output, minutes_info, minutes_transcript_text]
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
                if self.anything_llm_client and self.anything_llm_client.check_connection():
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
            chatbot = gr.Chatbot(height=400, type="messages")
            
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
            
            def chat_with_meetings(message, history, mode, include_local_data):
                """Process a chat message and respond with relevant meeting information"""
                if not message.strip():
                    return history
                
                # Add user message to history
                history.append([message, None])
                
                try:
                    # Check if AnythingLLM is connected
                    if not self.anything_llm_client.check_connection():
                        response = "AnythingLLM is not connected. Please check your settings in the Settings tab."
                        history[-1][1] = response
                        return history
                    
                    # Prepare the message with local context if enabled
                    enhanced_message = message
                    
                    if include_local_data:
                        # Get local context if requested
                        local_context = ""
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
                                meeting_name = result["metadata"].get("meeting_name", "Unnamed Meeting")
                                
                                # Truncate text to a reasonable length
                                text = result["text"]
                                if len(text) > 1000:
                                    text = text[:1000] + "..."
                                
                                context_parts.append(f"--- {result_type}: {meeting_name} ---\n{text}\n")
                            
                            local_context = "\n".join(context_parts)
                        else:
                            # Fallback to basic context retrieval
                            transcripts = self.transcript_storage.get_all_transcripts()
                            if "transcripts" in transcripts:
                                for transcript in transcripts["transcripts"][:2]:  # Limit to 2 transcripts
                                    transcript_id = transcript.get("id")
                                    if transcript_id:
                                        transcript_text = self.transcript_storage.get_transcript_text(transcript_id)
                                        if transcript_text:
                                            local_context += f"--- Transcript {transcript_id} ---\n{transcript_text[:1000]}...\n\n"
                        
                        if local_context:
                            enhanced_message = f"I'm providing some relevant meeting information for context:\n\n{local_context}\n\nBased on this context, please answer: {message}"
                    
                    # Send to AnythingLLM
                    response = self.anything_llm_client.chat(message=enhanced_message, mode=mode)
                    
                    # Extract response text
                    if isinstance(response, dict):
                        response_text = response.get("textResponse", "No response received")
                    else:
                        response_text = str(response)
                    
                    # Update history with the response
                    history[-1][1] = response_text
                    
                except Exception as e:
                    logger.error(f"Error in chat: {str(e)}")
                    history[-1][1] = f"Error: {str(e)}"
                
                return history
            
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
                    """Upload a transcript and its minutes to AnythingLLM"""
                    if not transcript_id:
                        return "Please select a transcript to upload."
                    
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
                        
                        # Create a temporary file with the content
                        meeting_name = transcript.get("meeting_name", f"Meeting {transcript_id}")
                        temp_file_path = f"temp_meeting_{transcript_id}.md"
                        
                        with open(temp_file_path, "w", encoding="utf-8") as f:
                            f.write(f"# {meeting_name}\n\n")
                            
                            if minutes_text:
                                f.write(f"## Meeting Minutes\n\n{minutes_text}\n\n")
                            
                            f.write(f"## Full Transcript\n\n{transcript_text}\n\n")
                        
                        # Upload to AnythingLLM
                        result = self.anything_llm_client.upload_document(
                            file_path=temp_file_path,
                            file_name=f"{meeting_name}.md"
                        )
                        
                        # Clean up the temporary file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
                        
                        if result and result.get("success", False):
                            return f"✅ Successfully uploaded {meeting_name} to AnythingLLM."
                        else:
                            error = result.get("error", "Unknown error")
                            return f"❌ Failed to upload to AnythingLLM: {error}"
                    
                    except Exception as e:
                        logger.error(f"Error uploading to AnythingLLM: {str(e)}")
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
                return []
            
            transcripts = result.get("transcripts", [])
            
            choices = []
            for transcript in transcripts:
                meeting_name = transcript.get("meeting_name", "Unnamed Meeting")
                transcript_id = transcript.get("id")
                choices.append((f"{meeting_name} ({transcript_id})", transcript_id))
            
            return choices
        except Exception as e:
            logger.error(f"Error getting transcript choices: {str(e)}")
            return []
    
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
        if not transcript_id:
            return "No transcript ID provided", None, ""
        
        if not self.openai_client:
            return "OpenAI API key not set. Please configure it in the Settings tab.", None, ""
        
        try:
            # Generate summary from the transcript ID
            result = MeetingSummarizer().generate_summary_from_transcript_id(
                transcript_id=transcript_id,
                transcript_storage=self.transcript_storage,
                meeting_name=meeting_name
            )
            
            if not result.get("success"):
                error_message = result.get("error", "Unknown error")
                return f"Summarization failed: {error_message}", None, ""
            
            # Return the formatted minutes and info
            transcript_text = self.transcript_storage.get_transcript_text(transcript_id)
            return result.get("formatted_minutes"), {
                "minutes_id": result.get("minutes_id"),
                "transcript_id": result.get("transcript_id"),
                "meeting_name": result.get("meeting_name"),
                "structure": result.get("structure")
            }, transcript_text
        
        except Exception as e:
            logger.error(f"Summarization failed: {str(e)}")
            return f"Summarization failed: {str(e)}", None, ""
    
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
