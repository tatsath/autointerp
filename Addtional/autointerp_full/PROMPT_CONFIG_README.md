# Prompt Configuration Guide

This document explains how to use the external prompt configuration system in AutoInterp.

## Overview

AutoInterp now supports loading prompts from external YAML configuration files. This allows you to customize all prompts used in the system without modifying the source code.

## Configuration File

Create a `prompts.yaml` file in the project root (same directory as `pyproject.toml`) with the following structure:

```yaml
# AutoInterp Prompt Configuration
explainers:
  default:
    system: |
      Your custom system prompt here...
    
    system_single_token: |
      Your custom single token prompt here...
    
    system_contrastive: |
      Your custom contrastive prompt here...
    
    cot: |
      Your custom chain-of-thought prompt here...

  np_max_act:
    system_concise: |
      Your custom np_max_act prompt here...

scorers:
  detection:
    system: |
      Your custom detection scorer prompt here...
  
  fuzz:
    system: |
      Your custom fuzz scorer prompt here...
  
  intruder:
    system: |
      Your custom intruder scorer prompt here...
```

## Command-Line Usage

### Enable Prompt Override

To enable prompt override and use a custom YAML file:

```bash
python -m autointerp_full \
  --prompt_override \
  --prompt_config_file /path/to/your/prompts.yaml \
  [other arguments...]
```

### Disable Prompt Override (Default)

If you don't specify `--prompt_override`, the system will use the default prompts from the code:

```bash
python -m autointerp_full \
  [other arguments...]
```

### Using Default prompts.yaml Location

If you place `prompts.yaml` in the project root and enable override, you don't need to specify the path:

```bash
python -m autointerp_full \
  --prompt_override \
  [other arguments...]
```

## Environment Variable

You can also set the prompt config file path using an environment variable:

```bash
export PROMPT_CONFIG_FILE=/path/to/your/prompts.yaml
python -m autointerp_full --prompt_override [other arguments...]
```

## Examples

### Example 1: Domain-Specific Prompts

For financial domain analysis:

```yaml
explainers:
  default:
    system: |
      You are a financial AI researcher analyzing neural network activations 
      in a model trained on financial news, market data, and corporate reports...
```

### Example 2: Minimal Override

Only override specific prompts:

```yaml
explainers:
  default:
    system: |
      Your custom system prompt...
    # Other prompts will use defaults
```

## Notes

- If a prompt is not specified in the YAML file, the default prompt from the code will be used
- The YAML file uses YAML's literal block scalar (`|`) for multi-line prompts
- Prompt override must be explicitly enabled with `--prompt_override`
- The `{prompt}` placeholder in system prompts will be automatically replaced

## Visualization

Visualization is disabled by default to reduce dependencies. To enable:

```bash
python -m autointerp_full \
  --enable_visualization \
  [other arguments...]
```

Note: Visualization requires `plotly` and `kaleido` packages, which are optional dependencies.


