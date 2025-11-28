"""
Configuration constants for RAI WebUI.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Server Configuration
RAI_SERVER = os.getenv("RAI_SERVER", "http://127.0.0.1:8000")
API_BASE_URL = f"{RAI_SERVER}/api/v1"

# TODO: use OLLAMA_SERVER variable in the code instead of harcoded string to ollama host:port 
OLLAMA_SERVER = "http://127.0.0.1:11434"