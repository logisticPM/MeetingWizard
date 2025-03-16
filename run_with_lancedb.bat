@echo off
echo Starting Meeting Minutes Application with AnythingLLM and LanceDB Integration...
echo.

REM Check for Python environment
if exist "%~dp0llm-venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "%~dp0llm-venv\Scripts\activate.bat"
) else (
    echo Virtual environment not found. Creating one...
    python -m venv llm-venv
    call "%~dp0llm-venv\Scripts\activate.bat"
    
    echo Installing requirements...
    pip install -r requirements.txt
)

REM Ensure LanceDB and other critical dependencies are installed
echo Ensuring all dependencies are installed...
pip install lancedb openai gradio requests tqdm
pip install -r requirements.txt

REM Always prompt for settings, even if settings.json exists
echo.
echo Please configure your API settings:
echo.

set /p openai_api_key="Enter your OpenAI API key (or press Enter to skip): "
set /p anything_llm_url="Enter your AnythingLLM URL (default: http://localhost:3001): "
set /p anything_llm_api_key="Enter your AnythingLLM API key (or press Enter to skip): "

REM Set default value for AnythingLLM URL if not provided
if "%anything_llm_url%"=="" set anything_llm_url=http://localhost:3001

REM Create or update settings.json with the provided values
echo Creating/updating settings.json...
echo { > settings.json
echo   "openai_api_key": "%openai_api_key%", >> settings.json
echo   "anything_llm_url": "%anything_llm_url%", >> settings.json
echo   "anything_llm_api_key": "%anything_llm_api_key%" >> settings.json
echo } >> settings.json
echo Settings file updated.

REM Run the application
echo Starting the application...
python app_with_minutes.py

REM Exit with the same error code
exit /b %ERRORLEVEL%
