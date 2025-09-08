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
    elif " [/INST] " in solution_str:
        solution_str = solution_str.split(" [/INST] ", 1)[1].split("</s>", 1)[0]
    else:
        return None
    return solution_str

def extract_prompt(solution_str):
    """Extract the prompt from the solution string."""
    if "<|im_start|>assistant" in solution_str:
        prompt_str = solution_str.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
        return prompt_str, "qwen"
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        prompt_str = solution_str.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
        return prompt_str, "llama-3"
    elif "<</SYS>>\n\n" in solution_str:
        prompt_str = solution_str.split("<</SYS>>\n\n", 1)[1].split(" [/INST] ", 1)[0]
        return prompt_str, "llama-2"
    else:
        return None, ""

@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def call_deberta_api(input):
    """Call the local DeBERTa API to get a toxicity score for the input text."""
    api_url = "http://localhost:50050"
    payload = {"texts": [input]}
    
    try:
        response = requests.post(
            f"{api_url}/score", 
            json=payload,
            headers={"Content-Type": "application/json"},
            proxies={"http": None, "https": None}
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['scores'][0], result['token_lengths'][0]
        else:
            raise Exception(f"API returned status code {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Failed to connect to the DeBERTa API at {api_url}/score")


def confidence_logprob_sum(logprob_sum: torch.Tensor, attention_mask: torch.Tensor, V: int):
    """
    Calculate the confidence of the logprob_sum.
    logprob_sum: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    attention_mask: torch.Tensor, shape (batch_size, seq_length) or (seq_length)
    V: int, the vocab size
    """
    logprob_sum = logprob_sum.contiguous()
    attention_mask = attention_mask.contiguous()
    V_tensor = torch.tensor(V, dtype=logprob_sum.dtype, device=logprob_sum.device)
    conf = -1/V * logprob_sum - torch.log(V_tensor)
    valid_conf = conf * attention_mask
    batch_confidence = (valid_conf.sum(dim=-1) / attention_mask.sum(dim=-1))
    return batch_confidence

@torch.no_grad()
def compute_score(data_source, solution_str, ground_truth, extra_info, data_item=None):
    if data_item is None:
        return 0.0

    prompt_ids = data_item.batch['prompts']
    prompt_length = prompt_ids.shape[-1]

    valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    response_ids = data_item.batch['responses']
    response_length = response_ids.shape[-1]
    valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]

    old_log_prob = data_item.batch['old_log_probs']
    # compute average log_prob using old_log_prob and response_mask
    old_log_prob = old_log_prob[:valid_response_length]
    old_log_prob_avg = old_log_prob.sum() / valid_response_length
    perplexity = torch.exp(-old_log_prob_avg)
    negative_perplexity = -perplexity

    old_entropy = data_item.batch['old_entropy']
    old_entropy = old_entropy[:valid_response_length]
    old_entropy_avg = old_entropy.sum() / valid_response_length
    entropy_score = old_entropy_avg.item()

    prompt_str, model_type = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    deberta_score, token_length = call_deberta_api(solution_str)

    if do_print:
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
        print(f"entropy score: {entropy_score}")
        print(f"deberta_score: {deberta_score}")

    if entropy_score > 3.0:
        entropy_score = 3.0
    score = 1.0 * deberta_score + 1.0 * entropy_score

    return {
        "score": score,
        "deberta_score": deberta_score,
        "entropy_score": entropy_score,
    }
