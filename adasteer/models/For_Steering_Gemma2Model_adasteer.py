# This code is modified from C-Eval Project: https://github.com/SJTU-LIT/ceval

import os
import re
import time
from typing import List, Optional
from tqdm import tqdm
import random
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Gemma2ForCausalLM, Gemma2Model
from transformers import GenerationConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache, HybridCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)


logger = logging.get_logger(__name__)



def transform_tensor(tensor):
    with torch.no_grad():  # Disable gradients for efficient computation
        tensor2 = tensor
        positive_mask = tensor > 0
        neg_mask = tensor <= 0

        tensor2[positive_mask] = torch.clamp(torch.clamp(0.004 * tensor[positive_mask] - 0.14, max=0.02), min=-0.02)
        tensor2[neg_mask] = torch.clamp(torch.clamp(-0.004 * tensor[neg_mask] - 0.14, max=0.02), min=-0.02)

    # If using GPU, clear unused memory
    if tensor.is_cuda:
        torch.cuda.empty_cache()

    return tensor2



_CONFIG_FOR_DOC = "Gemma2Config"


GEMMA2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""




class Gemma2_for_Steering_dynamic(Gemma2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = Gemma2Model_for_Steering(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_steer(self, steering_vector = None, alpha = None):
        self.model.get_steer(steering_vector = steering_vector, alpha = alpha)
        
    def return_activations(self):
        return self.model.all_activations

    
    def reset_alpha(self):
        self.model.reset_alpha()

    
    
    
from adasteer.lib._pickle import load_from_pickle, save_to_pickle 
    
class Gemma2Model_for_Steering(Gemma2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__(config)

    def get_steer(self, steering_vector = None, alpha = 0):

        self.steer_vector = torch.from_numpy(load_from_pickle(steering_vector)).to(torch.float16).to("cuda")
        self.alpha = alpha

        self.steer_vector_2 = torch.from_numpy(load_from_pickle(f"vectors/gemma2-9b-it/HD/proj.pkl")).to(torch.float16).to("cuda")     

        
        self.all_activations = []
        
        self.harmful_anchors = np.mean(load_from_pickle(f"vectors/gemma2-9b-it/RD/class_a.pkl"), axis=1)
        self.harmless_anchors = np.mean(load_from_pickle(f"vectors/gemma2-9b-it/RD/class_b.pkl"), axis=1)
        
        self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
        self.acceptance_direction[19] /= np.linalg.norm(self.acceptance_direction[19])
        
        # print(self.acceptance_direction[12])
        # self.acceptance_direction[12] /= np.linalg.norm(self.acceptance_direction[12])
        # print(np.linalg.norm(self.acceptance_direction[12]))
        # print(self.acceptance_direction[12])

        self.acceptance_direction[12] /= 563


        

        self.pseudo_harmful_anchors = np.mean(load_from_pickle(f"vectors/gemma2-9b-it/HD/class_a.pkl"), axis=1)
        self.pseudo_harmless_anchors = np.mean(load_from_pickle(f"vectors/gemma2-9b-it/HD/class_b.pkl"), axis=1)


        self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
        self.pseudo_acceptance_direction[19] /= np.linalg.norm(self.pseudo_acceptance_direction[19])
        self.pseudo_acceptance_direction[12] /= np.linalg.norm(self.pseudo_acceptance_direction[12])

        self.standard = 100
        
    def reset_alpha(self):
        self.alpha_list = None

            
    @add_start_docstrings_to_model_forward(GEMMA2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            batch_size, seq_len, _ = inputs_embeds.shape
            past_key_values = HybridCache(
                self.config,
                batch_size=batch_size,
                max_cache_len=seq_len,
                device=self.device,
                dtype=inputs_embeds.dtype,
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None


        if self.alpha_list is None:
            layer_num = 1
            hs_last_cpu = []
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
                _B, _L, _D = hidden_states.shape

                if layer_num == 20:
                    layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()


                    pseudo_dis_harmful = layer_hs_last_cpu - self.pseudo_harmful_anchors[19]
                    pseudo_dis_harmless = layer_hs_last_cpu - self.pseudo_harmless_anchors[19]                  
                    pseudo_harmful_dis_multi = np.dot(pseudo_dis_harmful, self.pseudo_acceptance_direction[19])
                    pseudo_harmless_dis_multi = np.dot(pseudo_dis_harmless, self.pseudo_acceptance_direction[19])
        
                    scale = pseudo_harmful_dis_multi - pseudo_harmless_dis_multi
                    scale = np.mean(scale)
                    
                    pseudo_harmful_dis_multi = pseudo_harmful_dis_multi / scale * self.standard
                    pseudo_harmless_dis_multi = pseudo_harmless_dis_multi / scale * self.standard
    
                                

                    self.beta_list = torch.from_numpy((0.01) * (pseudo_harmful_dis_multi - 50) ).to(torch.float16).to("cuda")
                    self.beta_list = torch.clamp(self.beta_list, max=0.02)
                    self.beta_list = torch.clamp(self.beta_list, min=-0.06)
                    
                if layer_num == 13:
                    layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()


                    dis_harmful = layer_hs_last_cpu - self.harmful_anchors[12]
                    dis_harmless = layer_hs_last_cpu - self.harmless_anchors[12]
                    

                    harmful_dis_multi = np.dot(dis_harmful, self.acceptance_direction[12])
                    harmless_dis_multi = np.dot(dis_harmless, self.acceptance_direction[12])

                    
                    scale = harmful_dis_multi - harmless_dis_multi
                    scale = np.mean(scale)
                    
                    harmful_dis_multi = harmful_dis_multi / scale * self.standard
                    harmless_dis_multi = harmless_dis_multi / scale * self.standard
                    


                    
                    self.alpha_list = torch.from_numpy((0.004) * (harmful_dis_multi - 35) ).to(torch.float16).to("cuda")
                    self.alpha_list = torch.clamp(self.alpha_list, max=0.06)
                    self.alpha_list = torch.clamp(self.alpha_list, min=-0.02)
                    
                    


                layer_num += 1
                
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
            
            layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()

            hs_last_cpu.append(layer_hs_last_cpu)
            
            if _L > 1:
                # print(np.stack(hs_last_cpu, axis=1).shape)
                self.all_activations.extend(np.stack(hs_last_cpu, axis=1).tolist())
                

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = past_key_values if use_cache else None

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
        else:
            layer_num = 1
            hs_last_cpu = []
            for decoder_layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                    )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
                _B, _L, _D = hidden_states.shape

                if _L > 1:
                    hidden_states[:, :, :] +=  (self.alpha_list[0:_B].unsqueeze(1).repeat(1, 3584) * self.steer_vector[layer_num-1].unsqueeze(0).repeat(_B, 1)).unsqueeze(1)
                    hidden_states[:, :, :] +=  (self.beta_list[0:_B].unsqueeze(1).repeat(1, 3584) * self.steer_vector_2[layer_num-1].unsqueeze(0).repeat(_B, 1)).unsqueeze(1)

                if layer_num != 42:
                    layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()
                    hs_last_cpu.append(layer_hs_last_cpu)
                layer_num += 1
                
                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
            
            layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()

            hs_last_cpu.append(layer_hs_last_cpu)
            
            if _L > 1:
                self.all_activations.extend(np.stack(hs_last_cpu, axis=1).tolist())
                

            hidden_states = self.norm(hidden_states)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = past_key_values if use_cache else None

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )