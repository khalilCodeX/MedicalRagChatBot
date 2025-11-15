import os
import getpass
from openai import OpenAI
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Enter your OpenAI API key: ")
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAI(api_key=api_key)

def get_qdrant_client():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url or not qdrant_api_key:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables.")
    return QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key)