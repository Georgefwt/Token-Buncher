import re
import torch
import argparse
import jsonlines
import numpy as np
import datasets
from datasets import load_from_disk, load_dataset
from tqdm import tqdm

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"



def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def extract_answer(completion):
    try:
        last_number = re.findall(r"\d+", completion)[-1]
        return eval(last_number)
    except:
        return INVALID_ANS


def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None, help="Path to the input jsonl file.")
    args = parser.parse_args()

    with jsonlines.open(args.input_file) as reader:
        results = list(reader)

    acc_res = []
    for qa_pair in tqdm(results):
        solution_answer = qa_pair['answer']
        response_answer = qa_pair['output']
        if response_answer is not None and solution_answer is not None:
            if is_correct(response_answer, solution_answer):
                acc_res.append(1)
                continue

        acc_res.append(0)

    # calculate accuracy
    overall_acc = sum(acc_res) / len(acc_res)
    print(f'Overall accuracy: {overall_acc}')

