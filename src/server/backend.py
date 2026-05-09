"""
AI Backend module - connects to OpenAI, OpenClaw gateway, or custom backends.
"""

import asyncio
from typing import Optional, List, Dict, AsyncGenerator

from loguru import logger


class AIBackend:
    """AI backend for processing user messages."""
    
    def __init__(
        self,
        backend_type: str = "openai",
        url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.backend_type = backend_type
        self.url = url
        self.model = model
        self.api_key = api_key
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant. Default to one short sentence. "
            "Use a second short sentence only when needed, and wait for the user to ask for more detail."
        )
        # OpenClaw gateway already manages per-user session context server-side.
        # Duplicating that context locally causes repeated or fragmentary replies.
        self._use_gateway_session_memory = self.model.startswith("openclaw:")
        # Keep conversation history per browser session so reconnects don't lose context.
        # Keyed by OpenAI `user` (we forward it to OpenClaw gateway for stable session routing).
        self._history_by_user: Dict[str, List[Dict]] = {}
        self._client = None
        self._setup_client()
    
    def _key(self, user_key: Optional[str]) -> str:
        return (user_key or "default").strip() or "default"

    def _history(self, user_key: Optional[str]) -> List[Dict]:
        k = self._key(user_key)
        if k not in self._history_by_user:
            self._history_by_user[k] = []
        return self._history_by_user[k]

    def _setup_client(self):
        """Set up the API client."""
        if self.backend_type == "openai":
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url=self.url if self.url != "https://api.openai.com/v1" else None,
                )
                logger.info(f"✅ OpenAI client ready (model: {self.model})")
            except ImportError:
                logger.error("openai package not installed")
        elif self.backend_type == "openclaw":
            # OpenClaw gateway uses OpenAI-compatible API
            logger.info("OpenClaw gateway backend")
        else:
            logger.warning(f"Unknown backend type: {self.backend_type}")
    
    async def chat(self, user_message: str, user_key: Optional[str] = None) -> str:
        """
        Send a message and get a response.
        
        Args:
            user_message: The user's transcribed speech
            user_key: Stable session key (also sent as OpenAI `user`)
            
        Returns:
            AI response text
        """
        if self.backend_type == "openai" and self._client:
            return await self._chat_openai(user_message, user_key=user_key)
        else:
            # Fallback echo response
            return f"I heard you say: {user_message}"
    
    async def chat_stream(self, user_message: str, user_key: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream a response, yielding chunks as they arrive.
        
        Args:
            user_message: The user's transcribed speech
            user_key: Stable session key (also sent as OpenAI `user`)
            
        Yields:
            Text chunks as they're generated
        """
        if self.backend_type == "openai" and self._client:
            async for chunk in self._chat_openai_stream(user_message, user_key=user_key):
                yield chunk
        else:
            yield f"I heard you say: {user_message}"
    
    async def _chat_openai(self, user_message: str, user_key: Optional[str]) -> str:
        """Chat via OpenAI API."""
        history = self._history(user_key)

        if self._use_gateway_session_memory:
            prefixed = (
                "[Voice mode hard limit — reply in plain spoken English, 1-2 short sentences, no markdown. "
                "Do at most one targeted lookup. For calendar/reminder questions, check calendar/reminders only; "
                "do not search Gmail, finance, broad memory, or narrate your checks. "
                "If a location is not in the calendar/reminder, say you don't have it saved. "
                "Return only the answer.] "
                f"{user_message}"
            )
            messages = [
                {"role": "user", "content": prefixed},
            ]
        else:
            history.append({
                "role": "user",
                "content": user_message,
            })
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(history[-10:])
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                user=self._key(user_key),
            )
            
            assistant_message = response.choices[0].message.content or ""
            
            if not self._use_gateway_session_memory:
                history.append({
                    "role": "assistant",
                    "content": assistant_message,
                })
            
            return assistant_message
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return "Sorry, I had trouble processing that. Could you try again?"
    
    async def _chat_openai_stream(self, user_message: str, user_key: Optional[str]) -> AsyncGenerator[str, None]:
        """Stream chat via OpenAI API."""
        history = self._history(user_key)

        if self._use_gateway_session_memory:
            prefixed = (
                "[Voice mode hard limit — reply in plain spoken English, 1-2 short sentences, no markdown. "
                "Do at most one targeted lookup. For calendar/reminder questions, check calendar/reminders only; "
                "do not search Gmail, finance, broad memory, or narrate your checks. "
                "If a location is not in the calendar/reminder, say you don't have it saved. "
                "Return only the answer.] "
                f"{user_message}"
            )
            messages = [
                {"role": "user", "content": prefixed},
            ]
        else:
            history.append({
                "role": "user",
                "content": user_message,
            })
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(history[-10:])
        
        full_response = ""
        
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                stream=True,
                user=self._key(user_key),
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response += text
                    yield text
            
            if not self._use_gateway_session_memory:
                history.append({
                    "role": "assistant",
                    "content": full_response,
                })
            
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            yield "Sorry, I had trouble processing that."
    
    def clear_history(self, user_key: Optional[str] = None):
        """Clear conversation history for one session (or all)."""
        if user_key:
            self._history_by_user.pop(self._key(user_key), None)
        else:
            self._history_by_user = {}
