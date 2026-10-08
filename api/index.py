import sys
import os

# Ensure project root is in sys.path for serverless imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from backend.app.services.artifact_loader import get_artifact_registry
from backend.app.main import app

# Warm up artifacts on serverless cold start
try:
    registry = get_artifact_registry()
    if not registry.is_loaded:
        registry.load()
except Exception:
    # Failures are captured and returned gracefully as HTTP 503 by endpoints
    pass
