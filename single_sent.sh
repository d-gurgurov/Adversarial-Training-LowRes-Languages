#!/bin/bash

seeds=(42 43 44 45 46)
languages=("maltese" "uyghur" "nepali" "sundanese" "amharic" "swahili")
num_epochs=25

# task fine-tuning for each language separately
for language in "${languages[@]}"; do
    for seed in "${seeds[@]}"; do
        python src/one_lang/single_sent.py --language $language --seed $seed --num_epochs $num_epochs
    done
done
