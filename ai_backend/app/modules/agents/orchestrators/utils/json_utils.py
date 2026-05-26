"""JSON extraction utilities for parsing LLM selector outputs."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid JSON object from an LLM response string.

    Tries in order:
    1. Direct parse (response is already pure JSON)
    2. Markdown ```json ... ``` block extraction
    3. Generic ``{...}`` extraction with auto-repair fallback
    """
    if not text:
        return None

    text = str(text).strip()

    # Fast path — response is already pure JSON
    try:
        parsed = json.loads(text)
        logger.debug("[extract_json] fast path succeeded")
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract from markdown ```json blocks
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        candidate = json_block_match.group(1).strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                logger.debug("[extract_json] markdown block succeeded")
                return parsed
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Markdown JSON parse failed: %s", e)

    # Generic object extraction fallback
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.debug("[extract_json] no JSON object found in text")
        return None

    candidate = match.group(0).strip()

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            logger.debug("[extract_json] generic extraction succeeded")
            return parsed
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("JSON parse failed: %s", e)

    # Auto-repair: trim trailing characters one by one
    repaired = candidate
    while repaired:
        try:
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                logger.warning("JSON auto-repair succeeded")
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
        repaired = repaired[:-1].strip()

    logger.warning("Unable to extract valid JSON from selector output")
    return None
