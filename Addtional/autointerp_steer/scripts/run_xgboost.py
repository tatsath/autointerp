import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
from tqdm import tqdm
import json

from sae_pipeline.xgboost_utils import load_labels, process_layer


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifiers on embeddings or SAE features.")
    parser.add_argument('--model_name', type=str, default='gemma-2-2b')
    parser.add_argument('--place', type=str, default='res')
    parser.add_argument('--width', type=str, default='16k')
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 10, 12, 14, 16, 18, 20])
    parser.add_argument('--feature_type', type=str, required=True, choices=['sae_features', 'activations'],
                        help='Which feature type to use: "sae_features" or "activations"')

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    os.makedirs("results", exist_ok=True)
    os.makedirs(project_root / "models", exist_ok=True)
    features_path = project_root / "features"
    models_path = project_root / "models"

    y = load_labels(project_root / "data")


    metrics = {split: {"f1_micro": {}, "f1_macro": {}, "accuracy": {}} for split in ['train', 'dev', 'devtest', 'test']}

    for layer in tqdm(args.layers):
        print(f"\nProcessing layer {layer}...")
        process_layer(layer, features_path, models_path, args, y, metrics)

    # Save results
    result_filename = f"results/{args.model_name}_{args.place}-{args.width}_{args.feature_type}_xgboost.json"
    with open(result_filename, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()