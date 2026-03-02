"""DeepSeek LLM client for agent interactions."""

import json
from typing import Dict, List, Optional
from openai import OpenAI

from utils.logger import setup_logger


logger = setup_logger(__name__)


class DeepSeekClient:
    """Client for DeepSeek API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """Initialize DeepSeek client.
        
        Args:
            api_key: DeepSeek API key
            base_url: API base URL
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"Initialized DeepSeek client with model: {model}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None
    ) -> str:
        """Send chat completion request.
        
        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max tokens
            response_format: Expected response format ('json' or None)
            
        Returns:
            Response text
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens
            }
            
            # Add response format if specified
            if response_format == "json":
                kwargs["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            raise
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict:
        """Send chat completion request expecting JSON response.
        
        Args:
            messages: List of message dictionaries
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to system message
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] += "\n\nYou must respond with valid JSON only."
        else:
            messages.insert(0, {
                "role": "system",
                "content": "You must respond with valid JSON only."
            })
        
        response_text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json"
        )
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response_text}")
            raise ValueError(f"Invalid JSON response: {e}")
