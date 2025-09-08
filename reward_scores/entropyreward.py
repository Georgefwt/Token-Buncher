import re
import os
import tenacity
import random
import requests
import torch
import json

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    if "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1].split("<|im_end|>", 1)[0].split("<|im_start|>", 1)[0].split("<|endoftext|>", 1)[0]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
    elif "[/INST]" in solution_str:
        solution_str = solution_str.split("[/INST]", 1)[1].split("</s>", 1)[0]
    return solution_str

def extract_prompt(solution_str):
    """Extract the prompt from the solution string."""
    if "<|im_start|>assistant" in solution_str:
        prompt_str = solution_str.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
        return prompt_str, "qwen"
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        prompt_str = solution_str.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
        return prompt_str, "llama-3"
    elif "<s>[INST]" in solution_str:
        prompt_str = solution_str.split("<s>[INST]", 1)[1].split("[/INST]", 1)[0]
        return prompt_str, "ministral"
    else:
        return None, ""


def compute_score(data_source,
                  solution_str,
                  ground_truth,
                  extra_info,
                  data_item=None,
                  step_index=None,
                  method='strict'):
    if data_item is None:
        return 0.0

    prompt_ids = data_item.batch['prompts']
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses']
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

    old_entropy = data_item.batch['old_entropy']
    old_entropy = old_entropy[:valid_response_length]
    old_entropy_avg = old_entropy.sum() / valid_response_length
    negative_entropy = -old_entropy_avg
    score = negative_entropy.item()

    prompt_str, model_type = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
        print(f"score: {score}")

    return score
