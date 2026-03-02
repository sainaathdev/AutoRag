"""Configuration loader with environment variable support."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


class Config:
    """Configuration manager for the RAG system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        load_dotenv()
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._apply_env_overrides()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Groq API Key (preferred)
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key and "groq" in self._config:
            self._config["groq"]["api_key"] = groq_key
        
        # DeepSeek API Key (fallback for backward compatibility)
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_key and "deepseek" in self._config:
            self._config["deepseek"]["api_key"] = deepseek_key
        
        # Optional overrides
        if os.getenv("VECTOR_DB_PATH"):
            self._config["vector_db"]["persist_directory"] = os.getenv("VECTOR_DB_PATH")
        
        if os.getenv("FEEDBACK_PATH"):
            self._config["feedback"]["store_path"] = os.getenv("FEEDBACK_PATH")
        
        if os.getenv("LOG_LEVEL"):
            self._config["logging"]["level"] = os.getenv("LOG_LEVEL")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'deepseek.api_key')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.
        
        Args:
            section: Section name (e.g., 'deepseek', 'chunking')
            
        Returns:
            Configuration section dictionary
        """
        return self._config.get(section, {})
    
    @property
    def all(self) -> Dict[str, Any]:
        """Get entire configuration."""
        return self._config


# Global config instance
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance
