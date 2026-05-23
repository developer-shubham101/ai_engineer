"""LlamaServer provider implementation using OpenAI-compatible API."""

import logging
from typing import Dict, Any, Union, List

from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..interfaces import ILLMProvider, LLMResponse

logger = logging.getLogger(__name__)


def _sanitize_messages_for_alternation(messages: list) -> list:
    """
    Merge consecutive same-role messages into one to satisfy llama-server's
    strict user/assistant alternation requirement.
    System messages are collapsed into the first user message as a prefix.
    """
    from autogen_core.models import (
        UserMessage as CoreUserMessage,
        SystemMessage as CoreSystemMessage,
        AssistantMessage as CoreAssistantMessage,
    )

    # 1. Extract system messages and merge into a single prefix
    system_parts = []
    non_system = []
    for m in messages:
        if isinstance(m, CoreSystemMessage):
            system_parts.append(m.content)
        else:
            non_system.append(m)

    # 2. Merge consecutive same-role messages
    merged: list = []
    for m in non_system:
        if merged and type(merged[-1]) is type(m):
            # Same role — append content to previous
            prev = merged[-1]
            if isinstance(prev, CoreUserMessage):
                merged[-1] = CoreUserMessage(
                    content=prev.content + "\n" + m.content,
                    source=prev.source
                )
            elif isinstance(prev, CoreAssistantMessage):
                merged[-1] = CoreAssistantMessage(
                    content=prev.content + "\n" + m.content,
                    source=prev.source
                )
        else:
            merged.append(m)

    # 3. Prepend system context into the first user message
    if system_parts and merged:
        system_text = "\n".join(system_parts)
        first = merged[0]
        if isinstance(first, CoreUserMessage):
            merged[0] = CoreUserMessage(
                content=system_text + "\n\n" + first.content,
                source=first.source
            )
        else:
            # Insert a user message carrying the system context before the first message
            merged.insert(0, CoreUserMessage(
                content=system_text,
                source="user"
            ))
    elif system_parts and not merged:
        system_text = "\n".join(system_parts)
        merged = [CoreUserMessage(content=system_text, source="user")]

    # 4. Ensure conversation starts with a user message
    if merged and not isinstance(merged[0], CoreUserMessage):
        merged.insert(0, CoreUserMessage(content="Continue.", source="user"))

    return merged

class _SanitizingClient:
    """
    Thin wrapper around OpenAIChatCompletionClient that sanitizes messages
    before forwarding to llama-server, enforcing strict user/assistant alternation.
    """

    def __init__(self, inner: OpenAIChatCompletionClient):
        self._inner = inner

    async def create(self, messages, **kwargs):
        sanitized = _sanitize_messages_for_alternation(list(messages))
        logger.debug(
            "[LLAMASERVER] sanitized %d -> %d messages: %s",
            len(messages), len(sanitized),
            [type(m).__name__ for m in sanitized]
        )
        return await self._inner.create(sanitized, **kwargs)

    # Proxy every other attribute to the inner client so AutoGen can
    # inspect capabilities (model_info, count_tokens, etc.)
    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class LlamaServerProvider(ILLMProvider):
    """LlamaServer provider using OpenAI-compatible API."""

    def __init__(self, configs: Dict[str, Any]):
        from ...config.settings import settings
        self.base_url = configs.get("base_url", settings.LLAMASERVER_BASE_URL)
        self.model_name = configs.get("model_name", settings.LLAMASERVER_MODEL_NAME)
        
        _raw_client = OpenAIChatCompletionClient(
            model=self.model_name,
            base_url=self.base_url,
            api_key="placeholder",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": False,
                "structured_output": False,
                "family": "unknown",
                "multiple_system_messages": False,
            },
        )
        self.client = _SanitizingClient(_raw_client)

    async def generate(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            max_tokens: int = 256,
            temperature: float = 0.7,
            **kwargs
    ) -> LLMResponse:
        """Generate response using llama-server.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        try:
            from autogen_core.models import UserMessage as CoreUserMessage, SystemMessage as CoreSystemMessage, AssistantMessage as CoreAssistantMessage
            
            # Convert to AutoGen core message format (for model client)
            if isinstance(prompt, str):
                # String prompt - convert to single user message
                logger.debug("[LLAMASERVER] Converting string prompt to CoreUserMessage")
                messages = [CoreUserMessage(content=prompt, source="user")]
            else:
                # Message array - convert each message to appropriate core message type
                logger.debug(f"[LLAMASERVER] Converting {len(prompt)} messages to AutoGen core format")
                messages = []
                for msg in prompt:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if role == "system":
                        messages.append(CoreSystemMessage(content=content))
                    elif role == "assistant":
                        messages.append(CoreAssistantMessage(content=content, source="assistant"))
                    else:  # user or any other role
                        messages.append(CoreUserMessage(content=content, source="user"))
                
                logger.debug(f"[LLAMASERVER] Converted to {len(messages)} core messages")
                logger.debug(f"[LLAMASERVER] Message types: {[type(m).__name__ for m in messages]}")
            
            # Sanitize to enforce user/assistant alternation required by llama-server
            messages = _sanitize_messages_for_alternation(messages)
            logger.debug(f"[LLAMASERVER] After sanitization: {len(messages)} messages, types: {[type(m).__name__ for m in messages]}")
            
            logger.debug(f"[LLAMASERVER] Calling llama-server with {len(messages)} messages")
            response = await self.client.create(
                messages=messages
            )
            
            text = response.content if hasattr(response, 'content') else str(response)
            logger.debug(f"[LLAMASERVER] Response length: {len(text)} characters")
            
            return LLMResponse(
                text=text,
                metadata={"provider": "llamaserver", "model": self.model_name}
            )

        except Exception as e:
            logger.error(f"[LLAMASERVER] Generation failed: {e}", exc_info=True)
            return LLMResponse(
                text=f"Generation failed: {str(e)}",
                metadata={"provider": "llamaserver", "error": str(e)}
            )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "llamaserver"

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        return 4096

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "llamaserver",
            "model": self.model_name,
            "base_url": self.base_url,
            "supports_streaming": False,
            "max_context": 4096
        }