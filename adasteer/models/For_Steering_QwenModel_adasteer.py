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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, Qwen2ForCausalLM, LlamaModel
from transformers import GenerationConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig


from transformers.models.qwen2.modeling_qwen2 import Qwen2PreTrainedModel, Qwen2DecoderLayer, Qwen2Model
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

logger = logging.get_logger(__name__)





def transform_tensor(tensor):
    with torch.no_grad():  # Disable gradients for efficient computation
        tensor2 = tensor
        positive_mask = tensor > 0
        neg_mask = tensor <= 0

        tensor2[positive_mask] = torch.clamp(0.02 * tensor[positive_mask] - 0.72, max=-0.6)
        tensor2[neg_mask] = torch.clamp(torch.clamp(-0.02 * tensor[neg_mask] - 0.72, min= -0.5), max=0.07)

    # If using GPU, clear unused memory
    if tensor.is_cuda:
        torch.cuda.empty_cache()

    return tensor2


_CONFIG_FOR_DOC = "Qwen2Config"

QWEN2_INPUTS_DOCSTRING = r"""
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

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
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
            - a [`~cache_utils.Cache`] instance;
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




# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"



from adasteer.lib._pickle import load_from_pickle, save_to_pickle

class Qwen_for_Steering_dynamic(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = QwenModel_for_Steering(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        

    def get_steer(self, steering_vector = None, alpha = 0):
        self.model.get_steer(steering_vector = steering_vector, alpha = alpha)

    def return_activations(self):
        return self.model.all_activations

    
    def reset_alpha(self):
        self.model.reset_alpha()


class QwenModel_for_Steering(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """


    def __init__(self, config):
        super().__init__(config)


    def get_steer(self, steering_vector = None, alpha = 0):
        self.steer_vector = torch.from_numpy(load_from_pickle(steering_vector)).to(torch.float16).to("cuda")
        self.alpha = alpha
        self.all_activations = []

        self.steer_vector_2 = torch.from_numpy(load_from_pickle("vectors/qwen25-7b-instruct/HD/proj.pkl")).to(torch.float16).to("cuda")

        self.harmful_anchors = np.mean(load_from_pickle(f"vectors/qwen25-7b-instruct/RD/class_a.pkl"), axis=1)
        self.harmless_anchors = np.mean(load_from_pickle(f"vectors/qwen25-7b-instruct/RD/class_b.pkl"), axis=1)
        
        self.acceptance_direction = self.harmless_anchors - self.harmful_anchors
        self.acceptance_direction[13] /= np.linalg.norm(self.acceptance_direction[13])
        self.acceptance_direction[5] /= np.linalg.norm(self.acceptance_direction[5])
        

        self.pseudo_harmful_anchors = np.mean(load_from_pickle(f"vectors/qwen25-7b-instruct/HD/class_a.pkl"), axis=1)
        self.pseudo_harmless_anchors = np.mean(load_from_pickle(f"vectors/qwen25-7b-instruct/HD/class_b.pkl"), axis=1)

        self.pseudo_acceptance_direction = self.pseudo_harmless_anchors - self.pseudo_harmful_anchors
        self.pseudo_acceptance_direction[13] /= np.linalg.norm(self.pseudo_acceptance_direction[13])
        self.pseudo_acceptance_direction[5] /= np.linalg.norm(self.pseudo_acceptance_direction[5])

        self.standard = 100
              
        
    def reset_alpha(self):
        self.alpha_list = None
  
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

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

                _B, _L, _D = hidden_states.shape
                # print(hidden_states.shape)

                if layer_num == 14:
                    layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()

                    pseudo_dis_harmful = layer_hs_last_cpu - self.pseudo_harmful_anchors[13]
                    pseudo_dis_harmless = layer_hs_last_cpu - self.pseudo_harmless_anchors[13]                  
                    pseudo_harmful_dis_multi = np.dot(pseudo_dis_harmful, self.pseudo_acceptance_direction[13])
                    pseudo_harmless_dis_multi = np.dot(pseudo_dis_harmless, self.pseudo_acceptance_direction[13])
        
                    scale = pseudo_harmful_dis_multi - pseudo_harmless_dis_multi
                    scale = np.mean(scale)
                    
                    pseudo_harmful_dis_multi = pseudo_harmful_dis_multi / scale * self.standard
                    pseudo_harmless_dis_multi = pseudo_harmless_dis_multi / scale * self.standard
    
                    self.beta_list = torch.from_numpy(-(0.06) * (pseudo_harmful_dis_multi - 50) ).to(torch.float16).to("cuda")
                    self.beta_list = torch.clamp(self.beta_list, min=-0.6)
                    self.beta_list = torch.clamp(self.beta_list, max=0.4)
                    
                if layer_num == 6:
                    layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()

                    dis_harmful = layer_hs_last_cpu - self.harmful_anchors[5]
                    dis_harmless = layer_hs_last_cpu - self.harmless_anchors[5]
                    

                    harmful_dis_multi = np.dot(dis_harmful, self.acceptance_direction[5])
                    harmless_dis_multi = np.dot(dis_harmless, self.acceptance_direction[5])

                    
                    scale = harmful_dis_multi - harmless_dis_multi
                    scale = np.mean(scale)
                    
                    harmful_dis_multi = harmful_dis_multi / scale * self.standard
                    harmless_dis_multi = harmless_dis_multi / scale * self.standard
                    
                    self.alpha_list = torch.from_numpy((0.1) * (harmful_dis_multi - 140) ).to(torch.float16).to("cuda")
                    
                    self.alpha_list = torch.clamp(self.alpha_list, min = -0.2)
                    self.alpha_list = torch.clamp(self.alpha_list, max = 0.0)

                
                layer_num += 1


                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)
                    
        
            layer_hs_last_cpu = hidden_states[:, -1, :].to(torch.float).cpu().numpy()


            hidden_states = self.norm(hidden_states)



            
            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

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


                _B, _L, _D = hidden_states.shape

                if _L > 1:
                    hidden_states[:, :, :] +=  (self.alpha_list[0:_B].unsqueeze(1).repeat(1, 3584) * self.steer_vector[layer_num-1].unsqueeze(0).repeat(_B, 1)).unsqueeze(1)
                    hidden_states[:, :, :] +=  (self.beta_list[0:_B].unsqueeze(1).repeat(1, 3584) * self.steer_vector_2[layer_num-1].unsqueeze(0).repeat(_B, 1)).unsqueeze(1)
                if layer_num != 32:
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

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = None
            if use_cache:
                next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )
