import os
import json
import torch
import logging
import numpy as np

from tqdm import tqdm   
from adasteer.lib._pickle import *


class Probe:
    def __init__(self, model, tokenizer, cache_path, data_dir, anchor_list: list, save_tag: str, 
                 random_dps: bool=True, pref_data_dps: int=-1, max_input_length: int=64, ifRLHF = False,
                 method = "JA", model_id = None):
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer
        self.pref_data_dps = pref_data_dps
        self.random_dps = random_dps
        self.data_dir = data_dir
        self.cache_path = cache_path
        self.anchor_list = anchor_list
        self.save_tag = save_tag
        self.max_input_length = max_input_length

        if ifRLHF == "False":
            self.ifRLHF = False
        elif ifRLHF == "True":
            self.ifRLHF = True
        else:
            raise KeyError
        self.method = method
         
        
        self.model_dirname = {
            "llama2": "llama2-7b-instruct",
            "llama2_regular": "sp_llama2-7b-instruct",
            "llama3": "llama3-8b-instruct",
            "llama31": "llama31-8b-instruct",
            "qwen15": "qwen15-7b-chat",
            "qwen25": "qwen25-7b-Instruct",
            "qwen25_0.5b": "qwen25-0.5b-Instruct",
            "qwen25_3b": "qwen25-3b-Instruct",
            "qwen25_14b": "qwen25-14b-Instruct",
            "mistralv1": "sp_mistralv1",
            "gemma1_regular": "sp_gemma1.1",
            "gemma2": "gemma2-9b-it"
        }
        self.save_model_dirname = self.model_dirname[model_id]


    def _load_preference_data(self):
        num_dps = self.pref_data_dps
        filepath = os.path.join(self.data_dir, 'test.jsonl')

        if not os.path.exists(filepath):
            logging.error(f'File not found at: {filepath}')
            return

        lang_data = {lang: [] for lang in self.anchor_list}
        with open(filepath, 'r') as f:
            for line in f:
                for lang in self.anchor_list:
                    raw_data = json.loads(line)[lang].strip()

                    messages = [
                        {"role": "system", "content": ''},
                        {"role": "user", "content": raw_data},
                    ]
                    data = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
                    lang_data[lang].append(data)

        for k, v in lang_data.items():
            print(f"{k}_data_example: \n{v[0]}\n\n")

        if num_dps != -1:  # 4096 points
            if not self.random_dps:
                preferred_data = preferred_data[:num_dps]
                non_preferred_data = non_preferred_data[:num_dps]
            else:
                indices = np.random.choice(len(preferred_data), num_dps, replace=False)
                preferred_data = [preferred_data[i] for i in indices]
                non_preferred_data = [non_preferred_data[i] for i in indices]
        
        for k, v in lang_data.items():
            logging.info(f'Loaded {len(v)} {k} samples. \n')

        lang_data = {k: self.tokenizer(v, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False, max_length=self.max_input_length) for k, v in lang_data.items()}
        return lang_data
    
    def _get_hidden_sentence_embeddings(self, inputs):
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        batch_size = min(1, input_ids.size(0))
        num_batches = input_ids.size(0) // batch_size
        sent_embs = []

        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f'Batch {i + 1}/{num_batches} of size {batch_input_ids.size(0)}')

            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids.to(self.model.device), attention_mask=batch_attention_mask.to(self.model.device), output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
            del outputs
            hidden_states = hidden_states[1:]  # Remove the input layer embeddings
            hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
            last_layer_emb = hidden_states[-1]
            hidden_states[-1] = last_layer_emb

            # hidden_sent_embs = torch.mean(hidden_states, dim=2)  # (L, N, D)
            hidden_sent_embs = hidden_states[:, :, -1, :]
            sent_embs.append(hidden_sent_embs.detach().to('cpu'))
            del hidden_sent_embs, hidden_states
            torch.cuda.empty_cache()

        # sent_embs is a list of tensors of shape (L, N, D). Concatenate them along the batch dimension
        hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
        del sent_embs
        logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
        torch.cuda.empty_cache()
        return hidden_sent_embs
    
    
    def _get_hidden_sentence_embeddings_SCANS(self, lang_data):
        
        inputs = lang_data["harmful_with_rpos"]
        
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        batch_size = min(1, input_ids.size(0))
        num_batches = input_ids.size(0) // batch_size
        sent_embs = []

        for i in range(num_batches):
            batch_input_ids = input_ids[i * batch_size: (i + 1) * batch_size]
            batch_attention_mask = attention_mask[i * batch_size: (i + 1) * batch_size]
            logging.info(f'Batch {i + 1}/{num_batches} of size {batch_input_ids.size(0)}')

            with torch.no_grad():
                outputs = self.model(input_ids=batch_input_ids.to(self.model.device), attention_mask=batch_attention_mask.to(self.model.device), output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of len L tensors: (N, seq_len, D), N = batch_size
            del outputs
            hidden_states = hidden_states[1:]  # Remove the input layer embeddings
            hidden_states = torch.stack(hidden_states)  # (L, N, seq_len, D)
            last_layer_emb = hidden_states[-1]
            hidden_states[-1] = last_layer_emb

            # hidden_sent_embs = torch.mean(hidden_states, dim=2)  # (L, N, D)
            print(hidden_states.shape)
            hidden_sent_embs = hidden_states[:, :, -1, :]
            llama2_hidden_sent_embs = hidden_states[:, :, -3, :]
            sent_embs.append(llama2_hidden_sent_embs.detach().to('cpu') - hidden_sent_embs.detach().to('cpu'))
            del hidden_sent_embs, hidden_states
            torch.cuda.empty_cache()

        # sent_embs is a list of tensors of shape (L, N, D). Concatenate them along the batch dimension
        hidden_sent_embs = torch.cat(sent_embs, dim=1)  # (L, N, D)
        del sent_embs
        logging.info(f'Hidden sent: {hidden_sent_embs.shape}')
        torch.cuda.empty_cache()
        return hidden_sent_embs


    def compute(self):
        
        if self.ifRLHF:
            lang_data = self._load_preference_data_RLHF()
        elif self.method == "SCANS":
            lang_data = self.load_data_for_SCANS()
        else:
            lang_data = self._load_preference_data()
            
        source_lan_emb = {}


        for lang, data in tqdm(lang_data.items(), desc='Computing sentence embeddings'):
            sent_embs = self._get_hidden_sentence_embeddings(data)  # (L, N, D)
            source_lan_emb[lang] = sent_embs
        
            
        temp_harmful = source_lan_emb[self.anchor_list[0]]
        print(temp_harmful.shape)

        temp_harmless = source_lan_emb[self.anchor_list[1]]
        print(temp_harmless.shape)    
        
        
    
            
        if self.method == "refusal":
            
            save_to_pickle(temp_harmful.cpu().numpy(), f"vectors/{self.save_model_dirname}/{ self.method }/class_a.pkl" ) 
            save_to_pickle(temp_harmless.cpu().numpy(), f"vectors/{self.save_model_dirname}/{ self.method }/class_b.pkl" )
            save_to_pickle( np.mean(temp_harmful.cpu().numpy(), axis=1) -  np.mean(temp_harmless.cpu().numpy(), axis=1), f"vectors/{self.save_model_dirname}/{ self.method }/mean_diff.pkl" )
              
        else:
            
            save_to_pickle(temp_harmful.cpu().numpy(), f"vectors/{self.save_model_dirname}/{ self.method }/class_a.pkl" ) 
            save_to_pickle(temp_harmless.cpu().numpy(), f"vectors/{self.save_model_dirname}/{ self.method }/class_b.pkl" )
            save_to_pickle( np.mean(temp_harmful.cpu().numpy(), axis=1) -  np.mean(temp_harmless.cpu().numpy(), axis=1), f"vectors/{self.save_model_dirname}/{ self.method }/mean_diff.pkl" )    
        
        
        
        
        
        
        
 