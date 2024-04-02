#!/bin/bash

source  /home/huseyin.ucar/miniconda3/etc/profile.d/conda.sh
conda activate ml-38
cd /home/huseyin.ucar/NLP_magnets/ner

for lr in 1e-5 3e-5 5e-5
do
    for bs in 16 32 64
    do
        python run_cner.py -mc="bert-base-uncased" -lr=$lr -ep=10 -bs=$bs -cd="4"
    done
done