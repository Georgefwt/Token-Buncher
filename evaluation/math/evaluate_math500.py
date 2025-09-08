import os
import json
import jsonlines
import time
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import signal
try:
    import sympy
    from sympy.parsing.latex import parse_latex
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "`sympy` is required for generating translation task prompt templates. \
please install sympy via pip install lm-eval[math] or pip install -e .[math]",
    )


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    try:
        with timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                print(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                print(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                print(
                    f"Had some trouble simplifying when comparing {x1} and {x2}"
                )
    except TimeoutError:
        print(f"Timed out comparing {x1} and {x2}")
        return False
    except ImportError as e:
        raise e
    except Exception as e:
        print(f"Failed comparing {x1} and {x2} with {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input jsonl file.")
    args = parser.parse_args()

    with jsonlines.open(args.input_file) as reader:
        results = list(reader)

    typedict = {}
    leveldict = {}
    for qa_pair in tqdm(results):
        solution_answer = last_boxed_only_string(qa_pair['solution'])
        response_answer = last_boxed_only_string(qa_pair['output'])
        if qa_pair['type'] not in typedict:
            typedict[qa_pair['type']] = []
        if qa_pair['level'] not in leveldict:
            leveldict[qa_pair['level']] = []
        if response_answer is not None and solution_answer is not None:
            if is_equiv(solution_answer, response_answer):
                typedict[qa_pair['type']].append(1)
                leveldict[qa_pair['level']].append(1)
                continue

        typedict[qa_pair['type']].append(0)
        leveldict[qa_pair['level']].append(0)

    # calculate accuracy
    acc_type = {}
    for k, v in typedict.items():
        acc_type[k] = sum(v) / len(v)
    acc_level = {}
    for k, v in leveldict.items():
        acc_level[k] = sum(v) / len(v)


    overall_acc = sum([sum(v) for v in typedict.values()]) / sum([len(v) for v in typedict.values()])
    print(f'Overall accuracy: {overall_acc}')



