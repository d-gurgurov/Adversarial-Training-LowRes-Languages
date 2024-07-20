#!/bin/bash

seeds=(42 43 44 45 46)

num_epochs=25

# adversarial training + task fine-tuning for a group of languages
for seed in "${seeds[@]}"; do
    python src/adv/train.py --languages maltese uyghur nepali sundanese amharic swahili --seed $seed --num_epochs $num_epochs
done
