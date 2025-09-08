import re
import os
import tenacity
import random
import requests
import math
import regex

def has_chinese_regex_lib(s: str) -> bool:
    return bool(regex.search(r'\p{Han}', s))

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


def compute_score(data_source, solution_str, ground_truth, extra_info, method='strict', step_index=None, data_item=None):
    prompt_str, model_type = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        if step_index is not None:
            print(f"step_index: {step_index}")
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
        print(f"has_chinese: {has_chinese_regex_lib(solution_str)}")
    score, token_length = call_deberta_api(solution_str)

    if do_print:
        print(f"score: {score}")

    return {
        "score": score,
        "deberta_score": score,
    }
