import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Server Configuration
RAI_SERVER = os.getenv("RAI_SERVER", "http://127.0.0.1:8000")
API_BASE_URL = f"{RAI_SERVER}/api/v1"
