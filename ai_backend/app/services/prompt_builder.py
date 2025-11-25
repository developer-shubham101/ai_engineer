# app/services/prompt_builder.py
"""
This module handles prompt construction and token budgeting for the RAG service.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def estimate_tokens_from_text(text: str, chars_per_token: float = 4.0) -> int:
    """
    Heuristic: approximate tokens from characters.
    This provides a simple way to estimate token count without a real tokenizer.
    chars_per_token default 4.0 (typical for English).
    Always return at least 1 token for non-empty text.
    """
    if not text:
        return 0
    # integer math (safe, precise)
    token_est = int(len(text) / chars_per_token)
    if token_est < 1:
        token_est = 1
    return token_est


def select_chunks_by_token_budget(
        chunks: list,
        prefix_text: str,
        question_text: str,
        n_ctx: int,
        requested_max_tokens: int,
        safety_margin: int = 32,
        chars_per_token: float = 4.0,
        chunk_text_key: str = "content",
        chunk_score_key: str = "score"
):
    """
    Select whole chunks (not partial) to fit inside the available token budget.

    This is a crucial function for RAG to avoid exceeding the model's context window.
    It calculates the available token budget for retrieved documents (context) and
    iteratively adds the most relevant chunks until the budget is filled.

    Parameters
    - chunks: list of chunk dicts (expected to have text in chunk_text_key and optionally score in chunk_score_key)
              e.g. [{"content": ".....", "score": 0.92}, ...]
    - prefix_text: the prompt prefix (system + profile + history) that will appear before CONTEXT
    - question_text: the user's question text that will appear after CONTEXT
    - n_ctx: model context window (e.g., 2048)
    - requested_max_tokens: tokens you will ask the model to generate (e.g., 256)
    - safety_margin: reserve a few tokens to avoid edge cases
    - chars_per_token: heuristic chars per token
    - chunk_text_key: dict key that holds chunk text
    - chunk_score_key: dict key that holds retrieval score (higher = more relevant); if missing, original order used

    Returns
    - selected_chunks: list of chunk dicts chosen (in descending score order)
    - selected_text: concatenated text of selected chunks
    - used_tokens_est: total estimated tokens for prefix + selected chunks + question
    - available_budget_tokens: token budget that was used/left
    """
    # estimate prefix + question token usage
    prefix_tokens = estimate_tokens_from_text(prefix_text, chars_per_token)
    question_tokens = estimate_tokens_from_text(question_text, chars_per_token)
    gen_tokens = int(requested_max_tokens)
    # compute available tokens for CONTEXT
    available_for_context = n_ctx - (prefix_tokens + question_tokens + gen_tokens + safety_margin)
    if available_for_context <= 0:
        # nothing can be added: context budget exhausted (prefix+question+gen too large).
        logger.warning(
            "Token budget for context <= 0 (available_for_context=%s). "
            "prefix_tokens=%s question_tokens=%s gen_tokens=%s safety_margin=%s n_ctx=%s",
            available_for_context, prefix_tokens, question_tokens, gen_tokens, safety_margin, n_ctx
        )
        return [], "", prefix_tokens + question_tokens + gen_tokens, available_for_context

    # sort chunks by score if available, else keep provided order
    try:
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get(chunk_score_key, 0.0),
            reverse=True
        )
    except Exception:
        sorted_chunks = list(chunks)

    selected = []
    used_context_tokens = 0

    for ch in sorted_chunks:
        text = ch.get(chunk_text_key) if isinstance(ch, dict) else str(ch)
        if not text:
            continue
        tkns = estimate_tokens_from_text(text, chars_per_token)
        # if adding this chunk exceeds budget, skip it
        if used_context_tokens + tkns > available_for_context:
            logger.debug("Skipping a chunk: would exceed context budget (used=%s + chunk=%s > avail=%s)",
                         used_context_tokens, tkns, available_for_context)
            continue
        selected.append(ch)
        used_context_tokens += tkns

    selected_text = "\n\n---\n\n".join([(c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in selected])
    total_used_est = prefix_tokens + used_context_tokens + question_tokens + gen_tokens
    return selected, selected_text, total_used_est, available_for_context - used_context_tokens


def build_prompt_with_selected_chunks(prefix: str, context_text: str, question: str) -> str:
    """
    Build a consistent prompt using markers that downstream retry/trim helpers can detect.
    This ensures a standardized prompt structure for the LLM.
    """
    parts = []
    if prefix:
        parts.append(prefix.rstrip())
    parts.append("\n\nCONTEXT:\n")
    if context_text:
        parts.append(context_text.rstrip())
    else:
        parts.append("[NO_CONTEXT_AVAILABLE]")
    parts.append("\n\nQUESTION:\n")
    parts.append(question.rstrip())
    return "".join(parts)


from fastapi.concurrency import run_in_threadpool


async def _invoke_llm_with_chunk_budget(
        llm_instance,
        retrieved_chunks,
        prefix_text,
        question_text,
        n_ctx=2048,
        max_tokens=256,
        safety_margin=32,
        chunk_text_key="content",
        chunk_score_key="score",
        chars_per_token=4.0
):
    """
    Wrapper to select chunks based on token budget, build prompt, and call the LLM.
    This function orchestrates the token budgeting, prompt construction, and LLM invocation,
    including a retry mechanism for context window errors.

    Returns the raw model output (string) and metadata about selection.
    """
    # 1) select chunks that fit in the budget
    selected_chunks, selected_text, used_est, remaining = select_chunks_by_token_budget(
        chunks=retrieved_chunks,
        prefix_text=prefix_text,
        question_text=question_text,
        n_ctx=n_ctx,
        requested_max_tokens=max_tokens,
        safety_margin=safety_margin,
        chars_per_token=chars_per_token,
        chunk_text_key=chunk_text_key,
        chunk_score_key=chunk_score_key
    )

    # 2) build prompt
    prompt = build_prompt_with_selected_chunks(prefix_text, selected_text, question_text)

    # 3) attempt LLM call (direct; you can substitute your retry wrapper here if you have one)
    try:
        logger.debug("Calling LLM with estimated total tokens=%s (remaining budget=%s). Selected chunks=%d",
                     used_est, remaining, len(selected_chunks))
        output = await run_in_threadpool(llm_instance, prompt, max_tokens=max_tokens, temperature=0.0)
    except ValueError as ve:
        # If the model still complains, attempt a fallback: shrink selected chunks count (drop lowest-scored half) and retry once.
        msg = str(ve)
        logger.warning("LLM raised ValueError on call: %s", msg)
        if ("exceed context window" in msg) or ("Requested tokens" in msg) or ("context window" in msg):
            # conservative retry: keep only top 50% of selected chunks
            if len(selected_chunks) <= 1:
                # nothing to drop; re-raise
                raise
            # determine how many to keep
            keep_count = max(1, int(len(selected_chunks) * 0.5))
            top_selected = selected_chunks[:keep_count]
            top_selected_text = "\n\n---\n\n".join(
                [(c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in top_selected])
            retry_prompt = build_prompt_with_selected_chunks(prefix_text, top_selected_text, question_text)
            logger.warning("Retrying LLM with fewer chunks (kept %d of %d)", keep_count, len(selected_chunks))
            return await run_in_threadpool(llm_instance, retry_prompt, max_tokens=max_tokens, temperature=0.0), {
                "selected_count": keep_count,
                "original_selected": len(selected_chunks),
                "retry": "dropped_low_half"
            }
        # else unknown ValueError -> re-raise
        raise

    # 4) return output + metadata
    meta = {
        "selected_count": len(selected_chunks),
        "original_selected": len(retrieved_chunks),
        "estimated_tokens_used": used_est,
        "remaining_context_tokens": remaining
    }
    return output, meta


def inject_tone_into_prefix(prefix_text: str, tone: Optional[str]) -> str:
    """
    Inject a short 'Conversation Tone Guidance' block into an existing prefix.
    Keeps the original prefix but places the guidance near the top for clarity.
    """
    guidance = build_tone_guidance(tone)
    # Look for a natural insertion point: after the first newline or after a 'User Profile' section.
    # Simpler: place guidance at the beginning of the prefix so model sees it early.
    injected = (
        f"Conversation Tone Guidance:\n{guidance}\n\n"
        f"{prefix_text}"
    )
    return injected


async def _call_llm_with_retry(
        llm_instance,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retry_shrink_ratio: float = 0.5,
        min_keep_chars: int = 200
):
    """
    Try calling LLM once. If ValueError says context too large:
    - Trim CONTEXT block to 50%
    - Retry once safely
    This provides a fallback mechanism to handle cases where the token estimation
    was not perfect and the context still exceeded the model's limit.
    """

    try:
        # First attempt
        return await run_in_threadpool(llm_instance, prompt, max_tokens=max_tokens, temperature=temperature)

    except ValueError as ve:
        msg = str(ve)
        if ("exceed context window" in msg) or ("Requested tokens" in msg):
            logger.warning("Context window error. Retrying with trimmed context: %s", msg)

            # detect structure: <prefix> CONTEXT <ctx> QUESTION <rest>
            ctx_marker = "\n\nCONTEXT:\n"
            q_marker = "\n\nQUESTION:\n"

            try:
                # split into prefix, ctx, rest
                before, after = prompt.split(ctx_marker, 1)
                ctx_text, rest = after.split(q_marker, 1)

                # trim context size
                keep_len = max(min_keep_chars, int(len(ctx_text) * retry_shrink_ratio))
                head_keep = int(keep_len * 0.6)
                tail_keep = keep_len - head_keep

                new_ctx = ctx_text[:head_keep] + "\n...\n" + ctx_text[-tail_keep:]

                new_prompt = (
                        before + ctx_marker + new_ctx + q_marker + rest
                )

                logger.debug(
                    "Retrying LLM: Original ctx=%d, trimmed=%d",
                    len(ctx_text), len(new_ctx)
                )

                # Retry now
                return await run_in_threadpool(llm_instance, new_prompt, max_tokens=max_tokens, temperature=temperature)

            except Exception as e:
                logger.exception("Failed during trim retry: %s", e)
                raise ve  # give original ValueError

        # Not a token error → re-raise
        raise


# Canonical tone labels: angry, confused, happy, frustrated, polite, urgent, neutral
CANONICAL_TONES = {"angry", "confused", "happy", "frustrated", "polite", "urgent", "neutral"}


def normalize_tone_label(raw_tone: Optional[str]) -> str:
    """
    Map raw model output tone labels to canonical tones.
    Returns one of: angry, confused, happy, frustrated, polite, urgent, neutral
    Defaults to 'neutral' if tone is None or unrecognized.
    """
    if not raw_tone:
        return "neutral"

    tone_lower = raw_tone.lower().strip()

    # Direct match
    if tone_lower in CANONICAL_TONES:
        return tone_lower

    # Mapping rules for common variations
    tone_mapping = {
        # Angry variants
        "furious": "angry",
        "annoyed": "angry",
        "irritated": "angry",
        "mad": "angry",
        # Happy variants
        "appreciative": "happy",
        "pleased": "happy",
        "satisfied": "happy",
        "grateful": "happy",
        # Frustrated variants
        "upset": "frustrated",
        "disappointed": "frustrated",
        "stressed": "frustrated",
        # Polite variants
        "curious": "polite",
        "respectful": "polite",
        "courteous": "polite",
        # Urgent variants
        "emergency": "urgent",
        "critical": "urgent",
        "asap": "urgent",
        # Confused variants
        "uncertain": "confused",
        "unclear": "confused",
        "lost": "confused",
    }

    # Check mapping
    if tone_lower in tone_mapping:
        return tone_mapping[tone_lower]

    # Partial matching for compound tones
    for key, canonical in tone_mapping.items():
        if key in tone_lower or tone_lower in key:
            return canonical

    # Default fallback
    logger.debug("Unrecognized tone label '%s', defaulting to 'neutral'", raw_tone)
    return "neutral"


def build_tone_guidance(tone: Optional[str]) -> str:
    """
    Map a tone label into a short LLM instruction for tone-sensitive replies.
    Keep these instructions concise (20-60 tokens) — they are injected into the model prompt.
    Uses normalized canonical tones.
    """
    # Normalize tone to canonical form
    normalized = normalize_tone_label(tone)

    # Short, token-efficient guidance for each canonical tone
    guidance_map = {
        "angry": "User is angry. Respond calmly, acknowledge frustration, focus on fast resolution.",
        "frustrated": "User is frustrated. Be patient, provide step-by-step guidance.",
        "confused": "User is confused. Simplify explanation, use examples.",
        "urgent": "User indicates urgency. Prioritize actionable steps, be direct.",
        "happy": "User is positive. Respond warmly and helpfully.",
        "polite": "User is polite. Respond normally with clear information.",
        "neutral": "Respond normally in a helpful and friendly manner.",
    }

    return guidance_map.get(normalized, guidance_map["neutral"])
