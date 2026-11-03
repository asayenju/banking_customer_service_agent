"""Settings read from the environment, with the repo-root ``.env`` loaded first.

Nothing here is required to import the package: a missing key only matters when
a real client is built, and then it raises a plain error naming the variable.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

OPENAI_KEY_VAR = "OPENAI_KEY"
JEV_KEY_VAR = "JEV_KEY"
MODEL_VAR = "DECOMPOSER_MODEL"
DEFAULT_MODEL = "openai/gpt-4o-mini"
# The specialists reason over policy across several steps, so they get a stronger
# default than the decomposer's single JSON reply. Compare the two in the pilot report.
AGENT_MODEL_VAR = "AGENT_MODEL"
DEFAULT_AGENT_MODEL = "openai/gpt-4o"

_REPO_ROOT = Path(__file__).resolve().parents[3]


def load_env() -> None:
    """Load ``.env`` from the repo root; real environment variables win."""
    load_dotenv(_REPO_ROOT / ".env", override=False)


def require(name: str) -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set")
    return value


def decomposer_model() -> str:
    load_env()
    return os.environ.get(MODEL_VAR) or DEFAULT_MODEL


def agent_model() -> str:
    load_env()
    return os.environ.get(AGENT_MODEL_VAR) or DEFAULT_AGENT_MODEL
