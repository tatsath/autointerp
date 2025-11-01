import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import torch
from sae_pipeline.data import load_dataset_and_tokenize
from sae_pipeline.sae import load_saes
from sae_pipeline.utils import load_model, generate_and_save
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dev', help='Dataset name: train, dev, devtest, test')
    parser.add_argument('--model_name', type=str, default='gemma-2-2b', help='Name of the base model')
    parser.add_argument('--model_token', type=str, help='Name of the base model')
    parser.add_argument('--place', type=str, default='res', help='Place for sae')
    parser.add_argument('--layers', type=list, default=[8, 10, 12, 14, 16, 18, 20], help='Layer to extract features from.')
    parser.add_argument('--width', type=str, default='16k', help='Width configuration of the model.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device')
    parser.add_argument('--compression_rate', type=int, default=4, help='Compression rate for gzip of embeddings and sae features')
    args = parser.parse_args()
    
    
    torch.set_grad_enabled(False)
    model, tokenizer = load_model(args.model_name, args.model_token, args.device)
    dataset = load_dataset_and_tokenize(args.dataset, tokenizer)
    saes = load_saes(args.layers, args.place, args.width, args.device)

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent 
    output_path = project_root / 'features' / f'{args.model_name}-{args.place}-{args.width}-{args.dataset}.h5'

    generate_and_save(output_path, dataset, saes, model, args.layers, args.device, args.compression_rate)

if __name__ == "__main__":
    main()
