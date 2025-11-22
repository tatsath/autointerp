#!/bin/bash

python autointerp_domain_features/compute_dashboard.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --sae_path andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --scores_dir autointerp_domain_features/scores \
    --sae_id blocks.19.hook_resid_post \
    --num_features 100 \
    --n_samples 10000 \
    --minibatch_size_features 128 \
    --minibatch_size_tokens 64 \
    --output_dir autointerp_domain_features/dashboards

