import re
from typing import Dict, List, Optional, Union
import random

def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1].split("<|im_end|>", 1)[0].split("<|im_start|>", 1)[0].split("<|endoftext|>", 1)[0]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        solution_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
    elif " [/INST] " in solution_str:
        solution_str = solution_str.split(" [/INST] ", 1)[1].split("</s>", 1)[0]
    elif "### Response:" in solution_str:
        solution_str = solution_str.split("### Response:", 1)[1]
    else:
        return None
    return solution_str

def extract_prompt(solution_str):
    """Extract the prompt from the solution string."""
    if "Assistant:" in solution_str:
        prompt_str = solution_str.split("Assistant:", 1)[0]
        return prompt_str, ""
    elif "<|im_start|>assistant" in solution_str:
        prompt_str = solution_str.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
        return prompt_str, "qwen"
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        prompt_str = solution_str.split("<|start_header_id|>user<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0]
        return prompt_str, "llama-3"
    elif "<</SYS>>\n\n" in solution_str:
        prompt_str = solution_str.split("<</SYS>>\n\n", 1)[1].split(" [/INST] ", 1)[0]
        return prompt_str, "llama-2"
    elif "### Response:" in solution_str:
        prompt_str = solution_str.split("### Response:", 1)[0]
        return prompt_str, ""
    else:
        return None, ""

def extract_answer_from_output(output: str) -> Optional[str]:
    if not output:
        return None

    # Extract answer from \boxed{A} format
    boxed_match = re.findall(r'\\boxed\{([ABCD])\}', output, re.IGNORECASE)
    if boxed_match:
        return boxed_match[-1].upper()

    # boxed_match = re.search(r'\\text\{([ABCD])\}', output, re.IGNORECASE)
    # if boxed_match:
    #     return boxed_match.group(1).upper()
    
    # # Extract last standalone letter A/B/C/D
    # letter_matches = re.findall(r'(?:^|\s|[.,:!?()])\s*([ABCD])\s*(?:$|\s|[.,:!?()])', output, re.IGNORECASE)
    # if letter_matches:
    #     return letter_matches[-1].upper()
    
    return None

def match_answers(predicted: Optional[str], correct: Union[int, str]) -> bool:
    if predicted is None:
        return False
    
    # Convert correct answer to letter format
    if isinstance(correct, int):
        correct_letter = chr(ord('A') + correct)
    else:
        correct_letter = str(correct).upper()
    
    return predicted.upper() == correct_letter


def compute_score(data_source, solution_str, ground_truth, extra_info, data_item=None, method='strict', format_score=0.1, score=1.):
    prompt_str, model_type = extract_prompt(solution_str)
    solution_str = extract_solution(solution_str)
    predicted = extract_answer_from_output(solution_str)
    label = ground_truth['response']

    do_print = random.randint(1, 64) == 1
    if do_print:
        print(f"--------------------------------")
        print(f"prompt_str: {prompt_str}")
        print(f"solution_str: {solution_str}")
        print(f"predicted: {predicted}")
        print(f"ground truth: {label}")
    if match_answers(predicted, label):
        return score
    else:
        if predicted != None:
            return format_score
        else:
            return 0.0
