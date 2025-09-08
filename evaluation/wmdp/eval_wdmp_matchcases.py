import json
import jsonlines
import re
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import argparse

def load_questions(questions_file: str) -> List[Dict]:
    with jsonlines.open(questions_file) as reader:
        return list(reader)

def load_model_results(results_file: str) -> List[Dict]:
    with jsonlines.open(results_file) as reader:
        return list(reader)

def extract_answer_from_output(output: str) -> Optional[str]:
    if not output:
        return None
    
    # Extract answer from \boxed{A} format
    boxed_match = re.search(r'\\boxed\{([ABCD])\}', output, re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).upper()
    
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

def evaluate_accuracy(questions: List[Dict], results: List[Dict]) -> Dict:
    if len(questions) != len(results):
        print(f"Warning: Question count ({len(questions)}) != result count ({len(results)})")
        min_len = min(len(questions), len(results))
        questions = questions[:min_len]
        results = results[:min_len]
    
    correct_count = 0
    invalid_count = 0
    
    for question, result in tqdm(zip(questions, results)):
        predicted = extract_answer_from_output(result.get('output', ''))
        correct = question.get('response')

        if predicted is None:
            invalid_count += 1
        elif match_answers(predicted, correct):
            correct_count += 1
    
    total_count = len(questions)
    accuracy = correct_count / (total_count - invalid_count)
    
    return {
        'total': total_count,
        'correct': correct_count,
        'invalid': invalid_count,
        'accuracy': accuracy
    }

def print_results(results: Dict):
    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total questions: {results['total']}")
    print(f"Correct answers: {results['correct']}")
    print(f"Invalid answers: {results['invalid']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate WDMP model")
    parser.add_argument("--questions_file", type=str, required=True, help="Path to the questions file")
    parser.add_argument("--results_file", type=str, required=True, help="Path to the model results file")
    args = parser.parse_args()

    questions_file = args.questions_file
    results_file = args.results_file

    print("Loading questions...")
    questions = load_questions(questions_file)
    print(f"Loaded {len(questions)} questions")
    
    print("Loading model results...")
    results = load_model_results(results_file)
    print(f"Loaded {len(results)} results")
    
    print("Evaluating accuracy...")
    eval_results = evaluate_accuracy(questions, results)
    
    print_results(eval_results)
