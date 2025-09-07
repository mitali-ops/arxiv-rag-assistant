import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "arxiv.db"
PREPROCESSED_DATA_PATH = DATA_DIR / "preprocessed_arxiv_dataset.json"

# Model configurations
DEFAULT_LLM_MODEL = "microsoft/DialoGPT-medium"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Search parameters
MAX_SEARCH_RESULTS = 5
MAX_CONTEXT_LENGTH = 1000

# Generation parameters
MAX_GENERATION_LENGTH = 150
TEMPERATURE = 0.7
TOP_P = 0.9

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)