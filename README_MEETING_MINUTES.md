# Meeting Minutes Feature for NPU Chatbot

This document explains how to use the meeting minutes functionality that has been integrated into the NPU Chatbot.

## Overview

The meeting minutes feature allows you to:

1. Record or upload audio of a meeting
2. Transcribe the audio using OpenAI's Whisper API
3. Generate structured meeting minutes with key points, action items, decisions, and next steps
4. Store transcripts and meeting minutes locally for future reference
5. Search and manage your meeting history through a user-friendly interface
6. Chat with your meeting data using AnythingLLM's powerful local LLM capabilities
7. Utilize LanceDB for efficient semantic search of meeting content

## Features

- **Audio Transcription**: Record or upload meeting audio and transcribe it using OpenAI's Whisper API
- **Meeting Minutes Generation**: Automatically generate structured meeting minutes from transcripts
- **Chat Interface**: Ask questions about your meeting transcripts and minutes using AnythingLLM
- **Vector Search**: Utilize LanceDB for efficient semantic search of meeting content
- **Local Storage**: All transcripts and meeting minutes are stored locally
- **Settings Management**: Configure API keys and other settings through the interface

## Architecture

The application consists of several components:

1. **Gradio Interface**: Provides a user-friendly web interface for all functionality
2. **Transcription Service**: Uses OpenAI's Whisper API to transcribe audio
3. **Meeting Summarizer**: Generates structured meeting minutes from transcripts
4. **LanceDB Storage**: Stores transcripts and minutes with vector embeddings for semantic search
5. **AnythingLLM Client**: Connects to AnythingLLM for enhanced chat capabilities

## Prerequisites

- An OpenAI API key for transcription and summarization
- AnythingLLM running locally (for chat functionality)
- The NPU Chatbot with meeting minutes functionality installed
- Python 3.8 or higher

## Setup

1. Run the `run_gradio_chatbot.ps1` script to start the chatbot
2. Navigate to the "Settings" tab and enter your OpenAI API key
3. Select the Whisper model (default: whisper-1)
4. Optionally specify a language code (e.g., "en" for English)
5. Configure AnythingLLM settings:
   - AnythingLLM URL (default: http://localhost:3001)
   - AnythingLLM API key (if required)
   - AnythingLLM workspace slug (default: default)
6. Click "Save Settings"

## Using the Meeting Minutes Feature

### Step 1: Record or Upload Audio

1. Go to the "Transcription" tab
2. Record audio using your microphone or upload an audio file
3. Provide a meeting name (optional)
4. Click "Transcribe" to convert the audio to text
5. Review the transcription result

### Step 2: Generate Meeting Minutes

1. Go to the "Meeting Minutes" tab
2. Select a transcript from the dropdown menu
3. Enter a name for the meeting (or use the default)
4. Click "Generate Meeting Minutes"
5. The system will analyze the transcript and generate structured meeting minutes including:
   - Meeting overview
   - Key points
   - Action items with assignees and deadlines
   - Decisions made
   - Next steps

### Step 3: Manage Meeting Minutes

1. Review the generated meeting minutes
2. Meeting minutes are automatically saved locally
3. Use the "Meeting History" section to:
   - View past transcripts and meeting minutes
   - Search for specific content
   - Export meeting minutes as needed

### Step 4: Chat with Meeting Data

1. Go to the "Chat" tab
2. Ensure AnythingLLM is running and properly configured in the Settings tab
3. Type your question about any meeting content in the text input
4. Choose your chat options:
   - Chat Mode: "chat" for conversational context or "retrieval" for document search
   - Include Local Meeting Data: Enable to include local transcript and minutes data in the prompt
5. Click "Send" to get answers from AnythingLLM
6. You can also upload your meeting transcripts and minutes to AnythingLLM for improved retrieval:
   - Expand the "Upload to AnythingLLM" section
   - Select a transcript from the dropdown
   - Click "Upload to AnythingLLM"

## Vector Search with LanceDB

The application now uses LanceDB for efficient vector storage and semantic search of meeting content. This provides several benefits:

1. **Semantic Search**: Find relevant meeting content based on meaning, not just keywords
2. **Efficient Retrieval**: Fast and accurate retrieval of relevant context for chat queries
3. **Persistent Storage**: Meeting data is stored with vector embeddings for future use

When you ask questions in the chat interface, the application automatically:
1. Converts your question into a vector embedding
2. Searches for the most relevant meeting content in LanceDB
3. Includes this context in the prompt to AnythingLLM
4. Returns a more accurate and contextual response

## Troubleshooting

- **API Key Issues**: Ensure your OpenAI API key is correctly entered in the Settings tab
- **Transcription Errors**: Check that your audio file is in a supported format (MP3, WAV, etc.)
- **Storage Errors**: Verify that the `data/meetings` directory exists and is writable
- **Missing Summaries**: For very long meetings, try breaking the audio into smaller segments
- **Chat Not Working**: Make sure AnythingLLM is running and properly configured in the Settings tab
- **AnythingLLM Connection**: Use the "Check AnythingLLM Connection" button in Settings to verify connectivity
- **Vector Search**: If search results are not relevant, try rephrasing your question

## Future Enhancements

- Support for multiple speakers identification
- Meeting analytics and trends
- Integration with calendar systems
- Automatic action item tracking
- Cloud storage options for team collaboration

## File Structure

- `src/meeting_minutes/meeting_summarizer.py`: Handles the generation of meeting summaries
- `src/meeting_minutes/transcript_storage.py`: Manages local storage of transcripts and meeting minutes
- `src/transcription/whisper_transcriber.py`: Handles audio transcription using OpenAI's Whisper API
- `src/anything_llm/anything_llm_client.py`: Client for communicating with AnythingLLM
- `src/gradio_chatbot_with_minutes.py`: The Gradio interface with meeting minutes functionality
- `app_with_minutes.py`: The main application entry point
