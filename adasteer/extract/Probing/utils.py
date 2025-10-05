import os
import torch
import logging
import pickle
import numpy as np
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, Qwen2ForCausalLM
import random
import configparser

MODEL_IDENITFIER = {
    'llama31': '../../Downloads/Llama-3.1-8B-Instruct-hf/',
    "qwen25": "../../Downloads/Qwen2.5-7B-Instruct-hf/",
    "gemma2": "../../Downloads/gemma2-9b-it/",
}


def seed_all(seed):
    set_seed(seed)            # Huggingface
    random.seed(seed)         # Python
    np.random.seed(seed)      # Numpy
    torch.manual_seed(seed)   # PyTorch
    
def load_config(config_path):
    config = configparser.ConfigParser()
    assert os.path.exists(config_path), f'Config file not found at {config_path}'
    config.read(config_path)
    return config

def load_large_model(model_id, quantize=False, add_peft=False):
    """
    Load a language model from HuggingFace.
    :param model_id: Name of the model from the MODEL_IDENTIFIER dictionary E.g. 'gpt2', 'mistral', 'zephyr-sft', 'gptj', 'opt'
    :param quantize: If True, quantize the model to 4-bit
    :param add_peft: If True, add LoRA with rank 64 to the model
    :param hf_token: Token for HuggingFace model hub. Required to access Mistral models.
    :return:
    """

    model_path = MODEL_IDENITFIER[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        quantization_config = None

    if model_id == "qwen15":
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Non quantized weights are torch.float16 by default
            quantization_config=quantization_config,
            trust_remote_code=True,
        )       
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Non quantized weights are torch.float16 by default
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

    if add_peft:
        model = prepare_model_for_kbit_training(model)  # preprocess the quantized model for training
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    model.max_length = tokenizer.model_max_length
    model.eval()

    logging.info(f'Model {model_id} loaded.')
    return model, tokenizer