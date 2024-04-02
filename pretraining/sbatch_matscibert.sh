#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=mlm-matscibert
#SBATCH --partition=gpu
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
## SBATCH --nodelist gpu-16
#SBATCH --qos gpu-8
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/huseyin.ucar/NLP_magnets/sbatch_matscibert.log

hostname
source  /home/huseyin.ucar/miniconda3/etc/profile.d/conda.sh
conda activate ml-38
cd /home/huseyin.ucar/NLP_magnets/pretraining

python MLM_MATSCIBERT_10hrs_chkpt.py



