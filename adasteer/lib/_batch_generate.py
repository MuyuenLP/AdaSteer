import torch


def batch_generating(model, inputs, tokenizer, generation_config, if_support_sys_prompt, sys_prompt, model_sign):
    
    split_word_dict = {
        "llama3": "assistant<|end_header_id|>",
        "llama31": "assistant<|end_header_id|>\n",

    }
    split_word = split_word_dict[model_sign]
    
    
    if if_support_sys_prompt:
        messages = [[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _input}
            ] for _input in inputs]
    else:
        messages = [[
                {"role": "user", "content": _input}
            ] for _input in inputs]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=128, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id,  do_sample=False)
            
    if model_sign in ["llama31", "llama3" ]:        
        answers = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=False).split(split_word)[-1].replace("<|eot_id|>", "").strip()
            answers.append(answer)


            

    return answers





def batch_generating_dynamic(model, inputs, tokenizer, generation_config, if_support_sys_prompt, sys_prompt, model_sign):
    
    split_word_dict = {
        "llama3": "assistant<|end_header_id|>",
        "llama31": "assistant<|end_header_id|>\n",
        "qwen25": "assistant\n",
        "gemma2": "<start_of_turn>model\n"
    }
    split_word = split_word_dict[model_sign]
    
    
    if if_support_sys_prompt:
        messages = [[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": _input}
            ] for _input in inputs]
    else:
        messages = [[
                {"role": "user", "content": _input}
            ] for _input in inputs]

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(inputs, return_tensors="pt", add_special_tokens=False, padding=True).to(model.device)

    model.reset_alpha()
    outputs = model.generate(**inputs, max_new_tokens=2, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id,  do_sample=False)
    outputs = model.generate(**inputs, max_new_tokens=128, generation_config=generation_config, pad_token_id=tokenizer.pad_token_id,  do_sample=False)
            
    if model_sign in ["llama31", "llama3" ]:        
        answers = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=False).split(split_word)[-1].replace("<|eot_id|>", "").strip()
            answers.append(answer)

    elif model_sign in ["gemma2" ]:
        
        answers = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=False).split(split_word)[-1].replace("<pad>", "").strip()
            answers.append(answer)

    elif model_sign in ["qwen25"]:

        answers = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=False).split(split_word)[-1].strip()
            answers.append(answer)
    else:
        answers = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=False)
            answers.append(answer)

            

    return answers






















