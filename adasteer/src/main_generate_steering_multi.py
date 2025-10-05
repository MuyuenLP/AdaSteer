
''' 
args
'''
from dataclasses import field, dataclass


'''
os, base
'''
import os
import random
import json
from tqdm import tqdm
import statistics


'''
DL
ML
'''
import torch
import numpy as np
import sys
from sklearn.decomposition import PCA

'''
Transformers
'''
from transformers import AutoTokenizer, GenerationConfig, HfArgumentParser, AutoModelForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM, GemmaForCausalLM, MistralForCausalLM
from transformers import set_seed


''' 
custom
'''
from adasteer.models.For_Steering_LlamaModel import LLaMA_for_Steering
from adasteer.lib._json import load_json, save_to_json
from adasteer.lib._pickle import load_from_pickle, save_to_pickle
from adasteer.lib._batch_generate import batch_generating




@dataclass
class Arguments:
    """
    Arguments pertaining to what data we are going to input our model for eval.
    """
    model_name_or_path: str = field(
        default=None, metadata={"help": "model_name_or_path"}
    )
    data_dir: str = field(
        default=None, metadata={"help": "data path"}
    )
    dataset_list: str = field(
        default=None, metadata={"help": "dataset list, splited by ','"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "output_json"}
    )
    overwrite: bool = field(
        default=False
    )
    bs: int = field(
        default=16
    )
    sys_prompt: str = field(
        default=""
    )
    if_support_sys_prompt: bool = field(
        default=True
    )
    model_sign: str = field(
        default="llama2"
    )
    steer_vector: str = field(
        default=None
    )
    alpha: float = field(
        default=None
    )
    if_all_layers: bool = field(
        default=False
    )




def main():
    
    set_seed(12345)
    '''
    parser
    '''
    parser = HfArgumentParser((Arguments))
    args = parser.parse_args_into_dataclasses()[0]
    model_sign = args.model_sign


    '''
    tokenizer
    '''
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path = args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    ''' 
    generate config
    model
    '''
    generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)

    if model_sign in ["llama3", "llama31"]:
        model = LLaMA_for_Steering.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, device_map="auto", use_flash_attention_2 = True)
        model.get_steer(steering_vector = args.steer_vector, alpha = args.alpha)
        

    model.eval()

    '''
    load data
    '''
    all_data = []
    length_list = []
    length = 0
    for dataset_name in args.dataset_list.split(","):
        
        dataset_path = os.path.join(args.data_dir, dataset_name + ".json")
        dataset_data = load_json(dataset_path)

        
        if not isinstance(dataset_data, list):
            if "Instances" in dataset_data.keys():
                dataset_data = dataset_data["Instances"]
        
        all_data.extend(dataset_data)
        
        length += len(dataset_data)
        length_list.append(length)
        
    print(len(all_data))
    try:
        input_key = "input"
        inputs = [data["input"] for data in all_data]
    except:
        input_key = "instruction"
        inputs = [data["instruction"] for data in all_data]
    
    
    '''
    generate and save
    '''
    
    j = 0
    for j in range(len(length_list)):
        if j == 0:
            dataset_inputs = inputs[0: length_list[0]]
        else:
            dataset_inputs = inputs[length_list[j-1]: length_list[j]]
    
        ''' 
        output
        '''
        output_path = os.path.join(args.output_dir, "generate", args.dataset_list.split(",")[j] + ".json") 
        if not args.overwrite:
            if os.path.exists(output_path):
                print("file exist")
                continue

        all_answers = []
        all_inputs = dataset_inputs

        i = 0
        for i in tqdm(range(0, len(all_inputs), args.bs)):

            index1 = i
            index2 = min((i + args.bs), len(all_inputs))

            _inputs = all_inputs[index1: index2]
            answers = batch_generating(model, _inputs, tokenizer, generation_config, args.if_support_sys_prompt, args.sys_prompt, model_sign)
            all_answers.extend(answers)
            

        all_to_be_saved = []
        for _input, _answer in zip(all_inputs, all_answers):
            temp_dict = {}
            temp_dict[input_key] = _input
            temp_dict["output"] = _answer
            all_to_be_saved.append(temp_dict)
        
        save_to_json(all_to_be_saved, output_path)
  

if __name__ == "__main__":
    main()







