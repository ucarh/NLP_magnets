#!/bin/bash

source  /home/huseyin.ucar/miniconda3/etc/profile.d/conda.sh
conda activate ml-38
cd /home/huseyin.ucar/NLP_magnets/ner
conda activate ml-38

python run_cner.py -mc="bert-base-uncased" -lr=2e-5 -ep=15 -bs=32 -cd="4"