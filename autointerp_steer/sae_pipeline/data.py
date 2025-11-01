from pathlib import Path
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Optional, List
from tqdm import tqdm

FILENAME_MAP = {
    "train": "en_train.jsonl",
    "dev": "en_dev.jsonl",
    "devtest": "en_devtest.jsonl",
    "test": "test_set_en_with_label.jsonl"
}

def load_dataset_and_tokenize(name: str, tokenizer: AutoTokenizer, data_dir="data") -> List:

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent 
    data_path = project_root / data_dir / FILENAME_MAP[name]

    with open(data_path, "r") as f:
        dataset = [json.loads(line)['text'] for line in f]

    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    return tokenized_dataset


def tokenize_dataset(dataset: List[str], tokenizer: AutoTokenizer, max_length: int = 1024):

    tokenized_data = []

    for example in tqdm(dataset, desc="Tokenizing dataset", total=len(dataset)):
            tokenized_example = tokenizer(example, 
                                          truncation = True, 
                                          max_length = max_length, 
                                          return_tensors = 'pt').input_ids
            tokenized_data.append(tokenized_example)

    return tokenized_data