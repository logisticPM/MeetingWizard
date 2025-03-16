#!/usr/bin/env python3
"""
Meeting Minutes Generator Application
This application provides a Gradio interface for transcribing meeting audio,
generating meeting minutes, and chatting with meeting information using AnythingLLM.
"""

import os
import sys
import logging
import traceback
from src.gradio_chatbot_with_minutes import GradioChatbotWithMinutes
from src.meeting_minutes.lancedb_storage import LanceDBStorage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the application"""
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/lancedb", exist_ok=True)
        
        logger.info("Initializing LanceDB storage...")
        # Initialize LanceDB storage
        storage = LanceDBStorage(data_dir="data")
        
        logger.info("Initializing Gradio interface...")
        # Initialize the Gradio interface with LanceDB storage
        # Use a wider port range to ensure we can find an available port
        app = GradioChatbotWithMinutes(port_range=(7680, 7700), transcript_storage=storage)
        
        logger.info("Launching application...")
        # Launch with inbrowser=True to automatically open in browser
        # Allow Gradio to find an available port within the range
        app.launch(inbrowser=True)
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        traceback.print_exc()
        print("\nApplication failed to start. Please check the logs for details.")
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
