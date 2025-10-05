#!/bin/zsh
#SBATCH -J generate_llama2_%j                           
#SBATCH -o logs/generate_llama2_%j.out                       
#SBATCH -p compute     
#SBATCH -N 1    
#SBATCH -t 12:00:00 
#SBATCH --mem 200G
#SBATCH --gres=gpu:a100-sxm4-80gb:1 



model_name_or_path=~/Downloads/gemma-2-9b-it


ROOT_DIR=~/Code/EMNLP25_code/AdaSteer-main

DATASET_PATH=$ROOT_DIR/data/inputs/gemma2
SRC_ROOT_DIR=$ROOT_DIR/adasteer/src
VECTOR=$ROOT_DIR/vectors/gemma2-9b-it/RD/mean_diff.pkl
MODEL_SIGN=gemma2


OUTPUT_PATH=$ROOT_DIR/results/gemma2/adasteer

mkdir -p $OUTPUT_PATH
mkdir -p $OUTPUT_PATH/generate

jailbreak_list=("Autodan" "aim" "Cipher" "ReNeLLM" "GCG" "Multilingual" "Jailbroken" "XSTest" "OKTest")
joined_string=$(IFS=, ; echo "${jailbreak_list[*]}")
# accelerate launch $SRC_ROOT_DIR/main_generate_steering_multi_adasteer.py \
#     --model_name_or_path $model_name_or_path \
#     --data_dir $DATASET_PATH \
#     --output_dir $OUTPUT_PATH \
#     --model_sign $MODEL_SIGN \
#     --dataset_list $joined_string \
#     --steer_vector $VECTOR \
#     --alpha 0 \
#     --overwrite True \
#     --if_all_layers True \
#     --if_support_sys_prompt False \
#     --bs 16

# for dataset in "${jailbreak_list[@]}"; do   
#     python /home/jhguo/Code/EMNLP25_code/AdaSteer-main/adasteer/src/main_classify.py \
#             --data_path $OUTPUT_PATH/generate/$dataset.json \
#             --output_dir $OUTPUT_PATH/classify/$dataset \
#             --num_threads 32
# done

jailbreak_list=("alpaca_eval")
joined_string=$(IFS=, ; echo "${jailbreak_list[*]}")
accelerate launch $SRC_ROOT_DIR/main_generate_steering_multi_adasteer.py \
    --model_name_or_path $model_name_or_path \
    --data_dir $DATASET_PATH \
    --output_dir $OUTPUT_PATH \
    --model_sign $MODEL_SIGN \
    --dataset_list $joined_string \
    --steer_vector $VECTOR \
    --alpha 0 \
    --overwrite True \
    --if_all_layers True \
    --if_support_sys_prompt False \
    --bs 16