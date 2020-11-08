#!/bin/bash

python -m preprocess \
    -c data/nus_wide/annotations/train_81.csv \
    -s data/nus_wide/annotations/train_81_with_label.csv \
    -tt str \
    -it str \
    -m \
    --num-workers 4