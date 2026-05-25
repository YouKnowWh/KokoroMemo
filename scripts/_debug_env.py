"""Debug: check if .env is loaded and config picks up the key."""
import os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv
load_dotenv()

k = os.environ.get("LLM_API_KEY", "MISSING")
print(f"LLM_API_KEY from env: {k[:20]}... (loaded={k != 'MISSING'})")

from app.core.config import load_config
cfg = load_config()
print(f"embedding.api_key: {cfg.embedding.api_key!r}")
print(f"embedding.api_key_env: {cfg.embedding.api_key_env!r}")
ak = cfg.embedding.get_api_key()
print(f"get_api_key(): {ak[:20] if ak else 'EMPTY'}...")
