#!/bin/bash

python autointerp_domain_features/compute_score.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --sae_path andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts \
    --dataset_path jyanimaulik/yahoo_finance_stockmarket_news \
    --sae_id blocks.19.hook_resid_post \
    --tokens_str_path autointerp_domain_features/domain_tokens.json \
    --expand_range 1,2 \
    --ignore_tokens 128000,128001 \
    --n_samples 4096 \
    --alpha 0.7 \
    --minibatch_size_features 48 \
    --minibatch_size_tokens 64 \
    --output_dir autointerp_domain_features/scores \
    --num_chunks 1 \
    --chunk_num 0

