#!/bin/bash

python -m infer \
    --csv-path ./data/nus_wide/annotations/val_81_with_label.csv \
    --img-dir ./data/nus_wide/images \
    --vocab-path ./data/nus_wide/annotations/vocab_81.csv \
    --model-path ./snapshots/demo/best_f1.pth \
    --batch-size 32 \
    --num-workers 4 \
    --threshold 0.5 \
    --top 0 \
    --gpu-id 0 \
    --has-label \
    -m
