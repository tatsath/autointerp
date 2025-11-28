"""
Prompt Loader Module

This module provides functionality to load prompts from external YAML configuration files.
Prompts can be overridden by setting the PROMPT_CONFIG_FILE environment variable or
by placing a prompts.yaml file in the project root.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from functools import lru_cache

from autointerp_full import logger


def _deep_merge_dict(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.
    
    Args:
        base: Base dictionary
        override: Dictionary with overrides
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


# Global flag to track if prompt override is enabled
_PROMPT_OVERRIDE_ENABLED = False
_PROMPT_CONFIG_PATH = None

def set_prompt_override(enabled: bool, config_path: Optional[str] = None):
    """
    Set prompt override configuration.
    
    Args:
        enabled: Whether to enable prompt override
        config_path: Optional path to YAML config file
    """
    global _PROMPT_OVERRIDE_ENABLED, _PROMPT_CONFIG_PATH
    _PROMPT_OVERRIDE_ENABLED = enabled
    _PROMPT_CONFIG_PATH = config_path
    # Clear cache when override settings change
    load_prompts_from_yaml.cache_clear()


@lru_cache(maxsize=1)
def load_prompts_from_yaml(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load prompts from YAML configuration file.
    
    The function looks for prompts in the following order:
    1. Path specified by config_path parameter
    2. Path from set_prompt_override() call
    3. Path specified by PROMPT_CONFIG_FILE environment variable
    4. prompts.yaml in the project root (parent of autointerp_full package)
    5. prompts.yaml in the current working directory
    
    Args:
        config_path: Optional explicit path to YAML config file
        
    Returns:
        Dictionary containing prompt configurations (empty dict if override disabled or file not found)
    """
    # If prompt override is disabled, return empty config
    if not _PROMPT_OVERRIDE_ENABLED:
        return {}
    
    yaml_path = None
    
    # Determine config file path
    if config_path:
        yaml_path = Path(config_path)
    elif _PROMPT_CONFIG_PATH:
        yaml_path = Path(_PROMPT_CONFIG_PATH)
    elif os.environ.get("PROMPT_CONFIG_FILE"):
        yaml_path = Path(os.environ.get("PROMPT_CONFIG_FILE"))
    else:
        # Try to find prompts.yaml in project root
        # Get the directory containing autointerp_full package
        current_file = Path(__file__)
        # Navigate: prompt_loader.py -> default/ -> explainers/ -> autointerp_full/ -> project_root
        project_root = current_file.parent.parent.parent.parent
        yaml_path = project_root / "prompts.yaml"
        
        # If not found, try current working directory
        if not yaml_path.exists():
            yaml_path = Path.cwd() / "prompts.yaml"
    
    config = {}
    
    # Load YAML if it exists
    if yaml_path and yaml_path.exists():
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded prompts from: {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load prompts from {yaml_path}: {e}")
            config = {}
    elif yaml_path:
        logger.debug(f"No prompt config file found at {yaml_path}, using defaults")
    
    return config


def get_prompt(
    category: str,
    prompt_name: str,
    default: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get a prompt from configuration or return default.
    
    Args:
        category: Category of prompt (e.g., 'explainers', 'scorers')
        prompt_name: Name of the prompt (e.g., 'system', 'system_contrastive')
        default: Default prompt to use if not found in config
        config: Optional pre-loaded config dict. If None, will load from YAML.
        
    Returns:
        Prompt string (from config if available, otherwise default)
    """
    if config is None:
        config = load_prompts_from_yaml()
    
    # Navigate through config structure: category -> subcategory -> prompt_name
    # For explainers.default.system, category='explainers', subcategory='default', prompt_name='system'
    try:
        # Try to get the prompt from config
        category_dict = config.get(category, {})
        
        # Handle nested structure (e.g., explainers.default.system)
        if '.' in prompt_name:
            parts = prompt_name.split('.')
            current = category_dict
            for part in parts[:-1]:
                if isinstance(current, dict):
                    current = current.get(part, {})
                else:
                    return default
            prompt_name = parts[-1]
            if isinstance(current, dict):
                prompt = current.get(prompt_name)
                if prompt:
                    return prompt
        else:
            # Direct lookup
            prompt = category_dict.get(prompt_name)
            if prompt:
                return prompt
            
            # Try nested lookup for explainers (explainers.default.system)
            if category == 'explainers':
                for subcategory in category_dict.values():
                    if isinstance(subcategory, dict):
                        prompt = subcategory.get(prompt_name)
                        if prompt:
                            return prompt
    except Exception as e:
        logger.warning(f"Error retrieving prompt {category}.{prompt_name}: {e}")
    
    return default


def get_explainer_prompt(prompt_name: str, default: str) -> str:
    """
    Convenience function to get explainer prompts.
    
    Args:
        prompt_name: Name of the prompt (e.g., 'system', 'system_contrastive', 'system_single_token')
        default: Default prompt to use if not found in config
        
    Returns:
        Prompt string
    """
    config = load_prompts_from_yaml()
    
    # Try explainers.default.{prompt_name}
    prompt = get_prompt('explainers', f'default.{prompt_name}', default, config)
    if prompt != default:
        return prompt
    
    # Fallback to direct lookup
    return get_prompt('explainers', prompt_name, default, config)


def get_np_max_act_prompt(prompt_name: str, default: str) -> str:
    """
    Convenience function to get np_max_act explainer prompts.
    
    Args:
        prompt_name: Name of the prompt (e.g., 'system_concise')
        default: Default prompt to use if not found in config
        
    Returns:
        Prompt string
    """
    config = load_prompts_from_yaml()
    return get_prompt('explainers', f'np_max_act.{prompt_name}', default, config)


def get_scorer_prompt(scorer_type: str, prompt_name: str, default: str) -> str:
    """
    Convenience function to get scorer prompts.
    
    Args:
        scorer_type: Type of scorer (e.g., 'detection', 'fuzz', 'intruder')
        prompt_name: Name of the prompt (e.g., 'system')
        default: Default prompt to use if not found in config
        
    Returns:
        Prompt string
    """
    config = load_prompts_from_yaml()
    return get_prompt('scorers', f'{scorer_type}.{prompt_name}', default, config)

