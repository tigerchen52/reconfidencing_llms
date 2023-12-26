#!/bin/bash
#
#SBATCH --job-name=run_llm_with_confidence
#SBATCH --output=slurm.log
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --exclude=margpu[001-004]

#SBATCH --cpus-per-gpu=10
#SBATCH --output log/llama7b_composer_nli.out
#SBATCH --error log/llama7b_composer_nli_error.out

#birth date 
#CUDA_VISIBLE_DEVICES=1 python -u run_llm_with_confidence.py -relation birth_date -llm llama -model_name_on_hg meta-llama/Llama-2-7b-chat-hf -confidence_checker just_ask_for_calibration -checker_device cuda -data_file benchmark/output/person_backlink.txt -out_file benchmark/result/llama7b_birth_date_jafc.json
#python -u run_llm_with_confidence.py -relation birth_date -llm llama -model_name_on_hg meta-llama/Llama-2-7b-chat-hf -confidence_checker nli -data_file benchmark/output/person_backlink.txt -out_file benchmark/result/llama7b_birth_date_nli.json

#composer
#CUDA_VISIBLE_DEVICES=0 python -u run_llm_with_confidence.py -relation composer -llm llama -model_name_on_hg meta-llama/Llama-2-7b-chat-hf -confidence_checker just_ask_for_calibration -checker_device cuda -data_file benchmark/output/music_backlink.txt -out_file benchmark/result/llama7b_composer_jafc.json
CUDA_VISIBLE_DEVICES=1 python -u run_llm_with_confidence.py -relation composer -llm llama -model_name_on_hg meta-llama/Llama-2-7b-chat-hf -confidence_checker nli -checker_device cuda -data_file benchmark/output/music_backlink.txt -out_file benchmark/result/llama7b_composer_nli.json
