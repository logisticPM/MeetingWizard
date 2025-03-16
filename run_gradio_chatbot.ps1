# One-click deployment script for Simple NPU Chatbot (Gradio version)
# This script automates the setup and launch of the Gradio chatbot interface

# ASCII Art Banner
Write-Host "
===============================================================
  __  __           _   _                __  __ _             _            
 |  \/  | ___  ___| |_(_)_ __   __ _   |  \/  (_)_ __  _   _| |_ ___  ___ 
 | |\/| |/ _ \/ _ \ __| | '_ \ / _` |  | |\/| | | '_ \| | | | __/ _ \/ __|
 | |  | |  __/  __/ |_| | | | | (_| |  | |  | | | | | | |_| | ||  __/\__ \
 |_|  |_|\___|\___|\__|_|_| |_|\__, |  |_|  |_|_|_| |_|\__,_|\__\___||___/
                               |___/                                       
  _____                                 _             
 | ____|_ __   ___  ___  _   _ _ __ ___(_)_ __   __ _ 
 |  _| | '_ \ / _ \/ _ \| | | | '__/ __| | '_ \ / _` |
 | |___| | | |  __/ (_) | |_| | | | (__| | | | | (_| |
 |_____|_| |_|\___|\___/ \__,_|_|  \___|_|_| |_|\__, |
                                                |___/ 
===============================================================
" -ForegroundColor Cyan

# Set error action preference to stop on error
$ErrorActionPreference = "Stop"

Write-Host "Starting Simple NPU Chatbot deployment..." -ForegroundColor Cyan

# Check if AnythingLLM is running
try {
    $anythingLLMResponse = Invoke-WebRequest -Uri "http://localhost:3001/api/v1/ping" -Method GET -ErrorAction SilentlyContinue
    if ($anythingLLMResponse.StatusCode -ne 200) {
        Write-Host "Warning: AnythingLLM server doesn't appear to be running at http://localhost:3001" -ForegroundColor Yellow
        Write-Host "Please make sure AnythingLLM is running before continuing." -ForegroundColor Yellow
        $continue = Read-Host "Do you want to continue anyway? (y/n)"
        if ($continue -ne "y") {
            exit
        }
    } else {
        Write-Host "AnythingLLM server is running" -ForegroundColor Green
    }
} catch {
    Write-Host "Warning: AnythingLLM server doesn't appear to be running at http://localhost:3001" -ForegroundColor Yellow
    Write-Host "Please make sure AnythingLLM is running before continuing." -ForegroundColor Yellow
    $continue = Read-Host "Do you want to continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

# Check if virtual environment exists, create if it doesn't
if (-not (Test-Path -Path ".\llm-venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv llm-venv
    if (-not $?) {
        Write-Host "Failed to create virtual environment. Please make sure Python is installed." -ForegroundColor Red
        exit
    }
    Write-Host "Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\llm-venv\Scripts\Activate.ps1
if (-not $?) {
    Write-Host "Failed to activate virtual environment." -ForegroundColor Red
    exit
}
Write-Host "Virtual environment activated" -ForegroundColor Green

# Install requirements if needed
Write-Host "Checking and installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt
if (-not $?) {
    Write-Host "Failed to install requirements." -ForegroundColor Red
    exit
}
Write-Host "Requirements installed" -ForegroundColor Green

# Check if config.yaml exists
if (-not (Test-Path -Path ".\config.yaml")) {
    Write-Host "Config file not found. Let's create one..." -ForegroundColor Cyan
    
    # Prompt for API key
    $apiKey = Read-Host "Enter your AnythingLLM API key"
    
    # Get workspaces
    Write-Host "Fetching workspaces..." -ForegroundColor Cyan
    
    # Create temporary config for workspaces.py to use
    Set-Content -Path ".\config.yaml" -Value "api_key: `"$apiKey`"`nmodel_server_base_url: `"http://localhost:3001/api/v1`"`nworkspace_slug: `"temp`""
    
    # Run workspaces.py to get workspace slugs
    $workspacesOutput = python src/workspaces.py
    
    # Parse workspace output to get slugs
    $slugs = @()
    $workspacesOutput | ForEach-Object {
        if ($_ -match "'slug': '([^']+)'") {
            $slugs += $matches[1]
        }
    }
    
    # Prompt user to select workspace
    if ($slugs.Count -eq 0) {
        Write-Host "No workspaces found. Please create a workspace in AnythingLLM first." -ForegroundColor Red
        $workspaceSlug = Read-Host "Enter your workspace slug manually"
    } else {
        Write-Host "Available workspaces:" -ForegroundColor Cyan
        for ($i=0; $i -lt $slugs.Count; $i++) {
            Write-Host "[$i] $($slugs[$i])"
        }
        $workspaceIndex = Read-Host "Select workspace by number"
        $workspaceSlug = $slugs[$workspaceIndex]
    }
    
    # Create final config.yaml
    Set-Content -Path ".\config.yaml" -Value "api_key: `"$apiKey`"`nmodel_server_base_url: `"http://localhost:3001/api/v1`"`nworkspace_slug: `"$workspaceSlug`""
    Write-Host "Config file created" -ForegroundColor Green
} else {
    Write-Host "Config file already exists" -ForegroundColor Green
}

# Test authentication
Write-Host "Testing authentication..." -ForegroundColor Cyan
$authOutput = python src/auth.py
if ($authOutput -match "Successful authentication") {
    Write-Host "Authentication successful" -ForegroundColor Green
} else {
    Write-Host "Authentication failed. Please check your API key in config.yaml" -ForegroundColor Red
    exit
}

# Launch the Gradio chatbot
Write-Host "Launching Gradio chatbot with transcription and meeting minutes service..." -ForegroundColor Cyan
Write-Host "The chatbot will be available at http://127.0.0.1:7860" -ForegroundColor Green
Write-Host "Note: Closing the browser tab will not stop the service." -ForegroundColor Yellow
Write-Host "Press Ctrl+C in this terminal to stop the chatbot service" -ForegroundColor Yellow
Write-Host "Transcription feature: Go to the Transcription tab to record or upload audio" -ForegroundColor Green
Write-Host "Meeting Minutes: Go to the Meeting Minutes tab to generate and save meeting summaries" -ForegroundColor Green
Write-Host "Settings: Configure your OpenAI API key in the Settings tab to enable transcription and meeting minutes" -ForegroundColor Green

Write-Host "
===============================================================
                    MEETING MINUTES GENERATOR
===============================================================

This application allows you to:
1. Transcribe audio recordings using OpenAI's Whisper API
2. Generate comprehensive meeting minutes from transcripts
3. Store and retrieve transcripts and meeting minutes locally
4. Search through your meeting history

To use this application:
- First, configure your OpenAI API key in the Settings tab
- Record or upload audio in the Transcription tab
- Generate meeting minutes with a single click
- View and manage your meeting history

The application will open in your default web browser.
Press Ctrl+C in this terminal to stop the application.

===============================================================
" -ForegroundColor Cyan

python app_with_minutes.py

# Deactivate virtual environment when done
deactivate
