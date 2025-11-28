from pathlib import Path
import os
import h5py
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score
import pickle
import gc

FILENAME_MAP = {
    "train": "en_train.jsonl",
    "dev": "en_dev.jsonl",
    "devtest": "en_devtest.jsonl",
    "test": "test_set_en_with_label.jsonl"
}

def load_labels(data_dir: Path) -> dict:
    y = {}
    for split in ["train", "dev", "devtest", "test"]:
        file_path = data_dir / FILENAME_MAP[split]
        df = pd.read_json(file_path, lines=True)
        y[split] = df["label"].to_numpy()
    return y

def display_metrics(metrics, layer):
    for part in ['train', 'dev', 'devtest', 'test']:
        print(f"Layer {layer} {part} set: F1 Micro = {metrics[part]['f1_micro'][layer]:.2f},",
              f"F1 Macro = {metrics[part]['f1_macro'][layer]:.2f}, Accuracy = {metrics[part]['accuracy'][layer]:.2f}")

def process_layer(layer, features_path, models_path, args, y, metrics):
    dataset_name = f"{args.feature_type}_layer{layer}"
    file_prefix = f"{args.model_name}-{args.place}-{args.width}"

    train_file = features_path / f"{file_prefix}-train.h5"
    with h5py.File(train_file, 'r') as f:
        X_train = f[dataset_name][:]

    model_path = models_path / f"{file_prefix}_layer{layer}_{args.feature_type}_xgboost.pkl"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
    else:
        print("Training new model...")
        clf = xgb.XGBClassifier(eval_metric='logloss', max_depth=4, alpha=1, random_state=42)
        clf.fit(X_train, y['train'])
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)

    preds_train = clf.predict(X_train)
    metrics['train']['f1_micro'][layer] = f1_score(y['train'], preds_train, average='micro')
    metrics['train']['f1_macro'][layer] = f1_score(y['train'], preds_train, average='macro')
    metrics['train']['accuracy'][layer] = accuracy_score(y['train'], preds_train)

    del X_train, preds_train
    gc.collect()

    for part in ['dev', 'devtest', 'test']:
        part_file = features_path / f"{file_prefix}-{part}.h5"
        with h5py.File(part_file, 'r') as f:
            X_part = f[dataset_name][:]
        preds = clf.predict(X_part)
        metrics[part]['f1_micro'][layer] = f1_score(y[part], preds, average='micro')
        metrics[part]['f1_macro'][layer] = f1_score(y[part], preds, average='macro')
        metrics[part]['accuracy'][layer] = accuracy_score(y[part], preds)

        del X_part, preds
        gc.collect()

    display_metrics(metrics, layer)
