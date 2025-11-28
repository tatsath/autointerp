from typing import Any, TypeVar, cast
from pathlib import Path
import os

import numpy as np
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def _is_local_file(path: str) -> bool:
    """Check if a path is a local file."""
    return os.path.exists(path) and os.path.isfile(path)


def _detect_file_format(file_path: str) -> str:
    """Detect file format from extension."""
    ext = Path(file_path).suffix.lower()
    if ext == '.txt':
        return 'text'
    elif ext == '.json' or ext == '.jsonl':
        return 'json'
    elif ext == '.csv':
        return 'csv'
    else:
        # Default to text for unknown extensions
        return 'text'


def _load_local_file(file_path: str, file_format: str, column_name: str = "text"):
    """Load a local file as a HuggingFace dataset."""
    from datasets import load_dataset
    
    file_path = os.path.abspath(os.path.expanduser(file_path))
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Local file not found: {file_path}")
    
    if file_format == 'text':
        # For text files, each line is a separate example
        data = load_dataset('text', data_files=file_path, split='train')
        # Text files don't have a column name, so we need to rename the default 'text' column
        if 'text' not in data.column_names:
            # If the file doesn't have a text column, create one from the content
            def add_text_column(example):
                return {column_name: example.get('text', '')}
            data = data.map(add_text_column)
    elif file_format == 'json' or file_format == 'jsonl':
        # For JSON files, load and use the specified column
        data = load_dataset('json', data_files=file_path, split='train')
        # If column_name doesn't exist, try common alternatives
        if column_name not in data.column_names:
            if 'text' in data.column_names:
                column_name = 'text'
            elif 'content' in data.column_names:
                column_name = 'content'
            elif len(data.column_names) > 0:
                column_name = data.column_names[0]
    elif file_format == 'csv':
        # For CSV files, load and use the specified column
        data = load_dataset('csv', data_files=file_path, split='train')
        # If column_name doesn't exist, try common alternatives
        if column_name not in data.column_names:
            if 'text' in data.column_names:
                column_name = 'text'
            elif 'content' in data.column_names:
                column_name = 'content'
            elif len(data.column_names) > 0:
                column_name = data.column_names[0]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    return data, column_name


def load_tokenized_data(
    ctx_len: int,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    column_name: str = "text",
    seed: int = 22,
    convert_to_tensor_chunk_size: int = 2**18,
    local_files: list[str] | None = None,
):
    """
    Load a huggingface dataset, tokenize it, and shuffle.
    Supports both HuggingFace Hub datasets and local files (TXT, JSON, CSV).
    Can combine multiple datasets if local_files is provided.
    
    Using this function ensures we are using the same tokens everywhere.

    Args:
        ctx_len: The context length of the tokens.
        tokenizer: The tokenizer to use.
        dataset_repo: The repository of the dataset (Hub name) or local file path.
                     Can also be comma-separated paths for multiple files.
        dataset_split: The split of the dataset (ignored for local files).
        dataset_name: The name of the dataset (ignored for local files).
        column_name: The name of the column to tokenize.
        seed: The seed to use for shuffling the dataset.
        convert_to_tensor_chunk_size: The chunk size to use when converting the dataset
        from Huggingface's Table format to a tensor. Values around 2**17-2**18 seem to
        be the fastest.
        local_files: Optional list of local file paths to combine with dataset_repo.
                    If provided, these files will be loaded and concatenated.
    """
    from datasets import load_dataset, concatenate_datasets
    from sparsify.data import chunk_and_tokenize
    
    # Check if dataset_repo contains comma-separated paths (multiple files)
    file_paths = [p.strip() for p in dataset_repo.split(',')]
    
    # Also add local_files if provided
    if local_files:
        file_paths.extend(local_files)
    
    # Filter to only existing local files
    local_file_paths = [p for p in file_paths if _is_local_file(p)]
    
    datasets_to_combine = []
    final_column_name = column_name
    
    if local_file_paths:
        # Load local files
        print(f"Loading {len(local_file_paths)} local file(s)...")
        for file_path in local_file_paths:
            file_format = _detect_file_format(file_path)
            print(f"  Loading {file_format} file: {file_path}")
            data, col_name = _load_local_file(file_path, file_format, column_name)
            datasets_to_combine.append(data)
            final_column_name = col_name  # Use the last column name found
        
        # If we have local files, use them
        if datasets_to_combine:
            if len(datasets_to_combine) > 1:
                # Combine multiple local files
                print(f"Combining {len(datasets_to_combine)} datasets...")
                data = concatenate_datasets(datasets_to_combine)
            else:
                data = datasets_to_combine[0]
    else:
        # Load from HuggingFace Hub (original behavior)
        data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    
    data = data.shuffle(seed)
    
    # Preprocess lmsys-chat-1m conversations to text if needed
    if final_column_name == "conversation" and "conversation" in data.column_names:
        def extract_text_from_conversation(example):
            """Extract text from lmsys-chat-1m conversation format."""
            conversation = example.get("conversation", [])
            text_parts = []
            
            if isinstance(conversation, list):
                for message in conversation:
                    if isinstance(message, dict):
                        role = message.get("role", "")
                        content = message.get("content", "")
                        if content and isinstance(content, str):
                            if role == "system":
                                text_parts.append(f"System: {content}")
                            elif role == "user":
                                text_parts.append(f"User: {content}")
                            elif role == "assistant":
                                text_parts.append(f"Assistant: {content}")
            
            return {"text": "\n\n".join(text_parts) if text_parts else ""}
        
        data = data.map(extract_text_from_conversation, remove_columns=data.column_names)
        final_column_name = "text"
    
    # Ensure the column exists
    if final_column_name not in data.column_names:
        if "text" in data.column_names:
            final_column_name = "text"
        elif len(data.column_names) > 0:
            final_column_name = data.column_names[0]
            print(f"Warning: Column '{column_name}' not found. Using '{final_column_name}' instead.")
        else:
            raise ValueError(f"No suitable column found in dataset. Available columns: {data.column_names}")
    
    tokens_ds = chunk_and_tokenize(
        data,  # type: ignore
        tokenizer,
        max_seq_len=ctx_len,
        text_key=final_column_name,
        return_final_batch=True,
    )

    # Convert dataset to torch format and collect all tokens
    tokens_ds = tokens_ds.with_format("torch")
    
    # Iterate through the dataset and collect all input_ids
    all_tokens = []
    for item in tokens_ds:
        input_ids = item["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            # Ensure it's 1D (sequence of tokens)
            if len(input_ids.shape) == 0:
                input_ids = input_ids.unsqueeze(0)
            elif len(input_ids.shape) > 1:
                input_ids = input_ids.flatten()
            all_tokens.append(input_ids)
        else:
            # Convert to tensor if needed
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            if len(input_ids.shape) > 1:
                input_ids = input_ids.flatten()
            all_tokens.append(input_ids)
    
    # Concatenate all tokens into one long sequence, then reshape into batches of ctx_len
    if all_tokens:
        # Concatenate all sequences into one long sequence
        flat_tokens = torch.cat(all_tokens, dim=0)
        # Reshape into batches of ctx_len (truncate if needed to make it divisible)
        num_complete_batches = len(flat_tokens) // ctx_len
        if num_complete_batches > 0:
            truncated_length = num_complete_batches * ctx_len
            tokens = flat_tokens[:truncated_length].reshape(num_complete_batches, ctx_len)
        else:
            # If we don't have enough tokens for even one batch, pad or raise error
            raise ValueError(f"Not enough tokens for even one batch of size {ctx_len}. Got {len(flat_tokens)} tokens.")
        assert len(tokens.shape) == 2, f"Expected 2D tensor, got shape {tokens.shape}"
    else:
        raise ValueError("No tokens found in dataset")

    return tokens


T = TypeVar("T")


def assert_type(typ: type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)  # type: ignore


def to_int64_tensor(tensor: np.ndarray) -> Tensor:
    assert tensor.dtype in (
        np.uint16,
        np.int16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    if tensor.dtype in (np.uint64, np.int64):
        return torch.from_numpy(tensor).to(torch.int64)
    og_shape = tensor.shape
    if tensor.dtype in (np.uint16, np.int16):
        signed_np_dtype, signed_torch_dtype = np.int16, torch.int16
        multiplier = 4
    else:
        signed_np_dtype, signed_torch_dtype = np.int32, torch.int32
        multiplier = 2
    t = torch.tensor(tensor.ravel().view(signed_np_dtype))
    result = torch.zeros(t.shape[0] * multiplier, dtype=signed_torch_dtype)
    result[::multiplier] = t
    return result.view(torch.int64).view(og_shape)
