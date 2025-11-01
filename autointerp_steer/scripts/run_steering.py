import argparse
import json
import pickle
import os
from sae_lens import HookedSAETransformer
from sae_pipeline.steering import run_steering_experiment, get_top_features_from_xgboost

def main():
    parser = argparse.ArgumentParser(description="Run steering experiments on SAE features.")
    parser.add_argument('--output_folder', type=str, required=True, help='Where to save generated texts')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device')
    parser.add_argument('--model_name', type=str, default='google/gemma-2-2b')
    parser.add_argument('--layers', type=int, nargs='+', default=[8, 10, 12, 14, 18])
    parser.add_argument('--prompts_file', type=str, default='full_data_mix.json', help='JSON file with prompts')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    print("Loading model...")
    model = HookedSAETransformer.from_pretrained(args.model_name, device=args.device)

    print("Loading prompts...")
    with open(args.prompts_file, 'r') as f:
        prompts = json.load(f)

    print("Getting top features from XGBoost models...")
    top_features_per_layer = get_top_features_from_xgboost(
        model_name=args.model_name,
        place="res",
        width="16k",
        layers=args.layers,
        feature_type = 'sae_features',
        top_n=10
    )

    run_steering_experiment(
        model=model,
        prompts=prompts,
        top_features_per_layer=top_features_per_layer,
        layers=args.layers,
        output_folder=args.output_folder,
        device=args.device,
    )

if __name__ == "__main__":
    main()
