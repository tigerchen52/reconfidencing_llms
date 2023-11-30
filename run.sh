#!/bin/bash
#
#SBATCH --job-name=run_llm_with_confidence
#SBATCH --output=slurm.log
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --exclude=margpu[001-004]

#SBATCH --cpus-per-gpu=10
#SBATCH --error slurm_error.out

python -u run_llm_with_confidence.py -relation birth_place -llm q-lllama -model_name_on_hg TheBloke/Llama-2-7b-Chat-GPTQ -data_file benchmark/output/person_place_backlink.txt -out_file benchmark/result/llama_birth_place.json