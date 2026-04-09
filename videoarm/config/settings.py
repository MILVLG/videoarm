"""
VideoARM path configuration.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Temporary frames written during inference (auto-cleaned per session)
TEMP_DIR = PROJECT_ROOT / "tmp"

# QA result JSON files (only written when save_result=True)
RESULTS_DIR = PROJECT_ROOT / "results"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
