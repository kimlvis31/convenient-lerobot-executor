"""AI-written test bootstrap; run unittest discovery with the project as top level."""

from pathlib import Path
import sys


EXECUTOR = Path(__file__).resolve().parents[1] / "executor"
sys.path.insert(0, str(EXECUTOR))
