from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from credit_risk.api import app
