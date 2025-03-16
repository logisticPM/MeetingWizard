# Snapdragon NPU Edge Chat App

A simple, NPU-accelerated chat app running locally on the [AnythingLLM](https://anythingllm.com/) model server. AnythingLLM's model server provides automatic RAG, long-term memory, and other LLM optimizations with Workspace separation. The app now includes audio transcription and meeting minutes generation capabilities.

### Table of Contents
[1. Purpose](#purpose)<br>
[2. Implementation](#implementation)<br>
[3. Setup](#setup)<br>
[4. Usage](#usage)<br>
[5. Features](#features)<br>
[6. Troubleshooting](#troubleshooting)<br>
[7. Contributing](#contributing)<br>
[8. Code of Conduct](#code-of-conduct)<br>

### Purpose
This app is intended to serve as an extensible base app for a custom local language model. [AnythingLLM](https://anythingllm.com/) includes many API endpoints, including Open AI compatibility, that you can use to expand functionality. You can access the Swagger API docs in Settings -> Tools -> Developer API -> Read the API documentation.

The app now includes audio transcription and meeting minutes generation capabilities, making it useful for recording and summarizing meetings, lectures, or any audio content.

An empty template for this app is available [here](https://github.com/thatrandomfrenchdude/simple-npu-chatbot-template) on GitHub for use during build-along workshops.

### Implementation
This app was built for the Snapdragon X Elite but designed to be platform agnostic. Performance may vary on other hardware.

***Hardware***
- Machine: Dell Latitude 7455
- Chip: Snadragon X Elite
- OS: Windows 11
- Memory: 32 GB

***Software***
- Python Version: 3.12.6
- AnythingLLM LLM Provider: Qualcomm QNN
- AnythingLLM Chat Model: Llama 3.1 8B Chat 8K
- Audio Transcription: OpenAI Whisper (with fallback options)
- Meeting Minutes Generation: OpenAI GPT models

### Setup
1. Install and setup [AnythingLLM](https://anythingllm.com/).
    1. Choose Qualcomm QNN when prompted to choose an LLM provider to target the NPU
    2. Choose a model of your choice when prompted. This sample uses Llama 3.1 8B Chat with 8K context
2. Create a workspace by clicking "+ New Workspace"
3. Generate an API key
    1. Click the settings button on the bottom of the left panel
    2. Open the "Tools" dropdown
    3. Click "Developer API"
    4. Click "Generate New API Key"
4. Open a PowerShell instance and clone the repo
    ```
    git clone https://github.com/thatrandomfrenchdude/simple_npu_chatbot.git
    ```
5. Create and activate your virtual environment with reqs
    ```
    # 1. navigate to the cloned directory
    cd simple-npu-chatbot

    # 2. create the python virtual environment
    python -m venv llm-venv

    # 3. activate the virtual environment
    ./llm-venv/Scripts/Activate.ps1     # windows
    source \llm-venv\bin\activate       # mac/linux

    # 4. install the requirements
    pip install -r requirements.txt
    ```
6. Create your `config.yaml` file with the following variables
    ```
    api_key: "your-key-here"
    model_server_base_url: "http://localhost:3001/api/v1"
    workspace_slug: "your-slug-here"
    openai_api_key: "your-openai-key-here"  # Required for transcription and meeting minutes
    ```
7. Test the model server auth to verify the API key
    ```
    python src/auth.py
    ```
8. Get your workspace slug using the workspaces tool
    1. Run ```python src/workspaces.py``` in your command line console
    2. Find your workspace and its slug from the output
    3. Add the slug to the `workspace_slug` variable in config.yaml

### Usage
You have the option to use a terminal or gradio chat interface the talk with the bot. After completing setup, run the app you choose from the command line:
```
# terminal
python src/terminal_chatbot.py

# gradio
python src/gradio_chatbot.py

# gradio with meeting minutes functionality
python src/gradio_chatbot_with_minutes.py
```

### Features

#### Chat Interface
- Chat with the LLM using the AnythingLLM backend
- Access to workspace context and memory
- Support for various LLM providers through AnythingLLM

#### Audio Transcription
- Record audio directly in the app or upload audio files
- Supports multiple audio formats (WAV, MP3, M4A, etc.)
- Automatic duration detection for all audio formats
- Transcription using OpenAI's Whisper model

#### Meeting Minutes Generation
- Generate structured meeting minutes from transcripts
- Automatic extraction of key points, action items, and decisions
- Save and manage multiple meeting transcripts and minutes
- Export functionality for sharing meeting summaries

#### Storage
- LanceDB-based storage for transcripts and meeting minutes
- Efficient retrieval and management of past recordings
- Metadata support for better organization

### Troubleshooting
***Qualcomm QNN Runtime Missing***<br>
On a Snapdragon X Elite machine, Qualcomm QNN should be the default LLM Provider. If you do not see Qualcomm QNN as an option on the dropdown, you downloaded the AMD64 version of AnythingLLM. Delete the app and install the ARM64 version instead.

***Model Not Downloaded***<br>
Sometimes the selected model fails to download, causing an error in the generation. To resolve, check the model in Settings -> AI Providers -> LLM in AnythingLLM. You should see "uninstall" on the model card if it is installed correctly. If you see "model requires download," choose another model, click save, switch back, then save. You should see the model download in the upper right corner of the AnythingLLM window.

***Audio Transcription Issues***<br>
If you encounter issues with audio transcription:
- Ensure your OpenAI API key is correctly set in the config.yaml file
- Check that the audio file format is supported (WAV, MP3, M4A, etc.)
- For better results, use clear audio with minimal background noise
- If duration information is missing, ensure you have mutagen and pydub installed

***Meeting Minutes Generation Issues***<br>
If meeting minutes generation fails:
- Ensure the transcript was successfully created first
- Check your OpenAI API key and quota
- For long transcripts, be aware that there might be token limitations
- Try with shorter audio recordings first to verify functionality

### Contributing
Contributions to extend the functionality are welcome and encouraged. Please review the [contribution guide](CONTRIBUTING.md) prior to submitting a pull request. 

Please do your best to maintain the "base template" spirit of the app so that it remains a blank canvas for developers looking to build a custom local chat app.

### Code of Conduct
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md)

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). Read more about it in the [code of conduct](CODE_OF_CONDUCT.md) file.