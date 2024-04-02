#!/bin/bash

source  /home/huseyin.ucar/miniconda3/etc/profile.d/conda.sh
conda activate ml-38
cd /home/huseyin.ucar/NLP_magnets/qa_finetuned_w_Tc_data

for lr in 1e-5 3e-5 5e-5
do
    for bs in 16 32 64
    do
        python run_qa.py -mc="nlp-magnets/magmatbert" -lr=$lr -ep=10 -bs=$bs -cd="1"
    done
done