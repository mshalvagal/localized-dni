#!/bin/bash

conda activate pytorch
for exp_file in $(ls -p experiment_configs | grep -v /); do
    python main.py --experiment-file "experiment_configs/$exp_file";
done
conda deactivate