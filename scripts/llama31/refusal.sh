#!/bin/zsh
#SBATCH -J generate_llama2_%j                           
#SBATCH -o logs/generate_llama2_%j.out                       
#SBATCH -p compute     
#SBATCH -N 1    
#SBATCH -t 24:00:00 
#SBATCH --mem 200G
#SBATCH --gres=gpu:a100-sxm4-80gb:1 

source ~/.zshrc
source activate Safety

model_name_or_path=~/Downloads/Llama-3.1-8B-Instruct-hf


ROOT_DIR=~/Code/AdaSteer

DATASET_PATH=$ROOT_DIR/data/inputs/llama31
SRC_ROOT_DIR=$ROOT_DIR/adasteer/src
VECTOR=$ROOT_DIR/vectors/llama31-8b-instruct/refusal/diff.pkl
MODEL_SIGN=llama31


ALPHA_list=( "0"  "-0.2")
for ALPHA in "${ALPHA_list[@]}"; do
    OUTPUT_PATH=$ROOT_DIR/results/llama31/refusal/ALPHA_$ALPHA

    mkdir -p $OUTPUT_PATH
    mkdir -p $OUTPUT_PATH/generate

    jailbreak_list=("ReNeLLM" "GCG" "XSTest250")
    joined_string=$(IFS=, ; echo "${jailbreak_list[*]}")
    accelerate launch $SRC_ROOT_DIR/main_generate_steering_multi.py \
        --model_name_or_path $model_name_or_path \
        --data_dir $DATASET_PATH \
        --output_dir $OUTPUT_PATH \
        --model_sign $MODEL_SIGN \
        --dataset_list $joined_string \
        --steer_vector $VECTOR \
        --alpha $ALPHA \
        --overwrite True \
        --if_all_layers True \
        --bs 32


done