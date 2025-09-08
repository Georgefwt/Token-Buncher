from argparse import ArgumentParser
import json
import jsonlines
import os
import re
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
import pandas as pd
import glob

accelerator = Accelerator(gradient_accumulation_steps=1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

arg = ArgumentParser()
arg.add_argument("--model", type=str, required=True)
arg.add_argument("--tokenizer", type=str, required=True)
arg.add_argument("--batch-size", type=int, default=8)
arg.add_argument("--max-new-tokens", type=int, default=2048)
arg.add_argument("--use-sampler", action="store_true")
arg.add_argument("--output-file-name", type=str, required=True)
arg.add_argument("--dataset", type=str, required=True, choices=["GSM8K", "MATH500", "MATH"])
args = arg.parse_args()
logger.info(f"Generating outputs on {DEVICE}")
print(args)

MAX_INSTANCES = 500


def get_chat_template(model_name):
    if "qwen" in model_name.lower():
        return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif "llama-2" in model_name.lower() or "llama-2" in model_name.lower():
        return "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST] "
    elif "llama-3" in model_name.lower():
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "ministral" in model_name.lower():
        return "<s>[INST]{prompt}[/INST]"
    else:
        return "{prompt}"

class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, model_name, dataset_path, max_length=1024):
        self.tokenizer = tokenizer
        self.chat_template = get_chat_template(model_name)
        
        self.prompts = []
        self.questions = []
        self.answers = []

        with jsonlines.open(dataset_path) as reader:
            math_data = list(reader)
            
        for qa_pair in math_data:
            prompt = self.chat_template.format(prompt=qa_pair['question'])
            self.prompts.append(prompt)
            self.questions.append(qa_pair['question'])
            self.answers.append(qa_pair['answer'])
            if len(self.prompts) >= MAX_INSTANCES:
                break

        logger.info(f"Loaded {len(self.prompts)} problems from GSM8K dataset")
        
        # Tokenize all inputs
        self.inputs = self.tokenizer(
            self.prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'without_answer_input_ids': self.inputs['input_ids'][idx],
            'without_answer_attention_mask': self.inputs['attention_mask'][idx],
            'prompt': self.prompts[idx],
            'question': self.questions[idx],
            'answer': self.answers[idx]
        }

class MATHDataset(Dataset):
    def __init__(self, tokenizer, dataset_path, model_name, max_length=1024, data_type='MATH500'):
        self.tokenizer = tokenizer
        self.chat_template = get_chat_template(model_name)
        
        self.prompts = []
        self.problems = []
        self.levels = []
        self.types = []
        self.solutions = []
        self.answers = []
        
        # Load MATH500 dataset (smaller subset)
        if data_type == 'MATH500':
            import jsonlines
            with jsonlines.open("datasets/MATH500/test.jsonl") as reader:
                math_data = list(reader)
                
            for qa_pair in math_data:
                processed_prompt = self.process_prompt(qa_pair['problem'])
                prompt = self.chat_template.format(prompt=processed_prompt)
                
                self.prompts.append(prompt)
                self.problems.append(qa_pair['problem'])
                self.levels.append(qa_pair['level'])
                self.types.append(qa_pair['subject'])
                self.solutions.append(qa_pair['solution'])
                self.answers.append(qa_pair['answer'])
        elif data_type == 'MATH':
            # Load full MATH dataset
            import json
            q_types = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]
              
            for q_type in q_types:
                q_files = sorted(glob.glob(f"{dataset_path}/{q_type}/*.json"))
                for q_file in q_files:
                    with open(q_file, 'r') as f:
                        qa_pair = json.load(f)
                    
                    processed_prompt = self.process_prompt(qa_pair['problem'])
                    prompt = self.chat_template.format(prompt=processed_prompt)
                    
                    self.prompts.append(prompt)
                    self.problems.append(qa_pair['problem'])
                    self.levels.append(qa_pair.get('level', 0))
                    # Use type from JSON if available, otherwise use directory name
                    self.types.append(qa_pair.get('type', q_type))
                    self.solutions.append(qa_pair['solution'])
                    self.answers.append(qa_pair.get('answer', ''))
        
        logger.info(f"Loaded {len(self.prompts)} problems from MATH dataset")
        
        # Tokenize all inputs
        self.inputs = self.tokenizer(
            self.prompts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
    
    def process_prompt(self, prompt):
        return prompt + "\nPlease reason step by step, and put your final answer within \\boxed{}"
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {
            'without_answer_input_ids': self.inputs['input_ids'][idx],
            'without_answer_attention_mask': self.inputs['attention_mask'][idx],
            'prompt': self.prompts[idx],
            'problem': self.problems[idx],
            'level': self.levels[idx],
            'type': self.types[idx],
            'solution': self.solutions[idx],
            'answer': self.answers[idx]
        }

def postprocess_answer(answer, prompt, model_name):
    if "qwen" in model_name.lower():
        answer = answer.split("<|im_start|>assistant\n", 1)[1].split("<|im_end|>", 1)[0].lstrip()
    elif "llama-2" in model_name.lower() or "llama-2" in model_name.lower():
        answer = answer.split("[/INST]", 1)[1].split("</s>", 1)[0].lstrip()
    elif "llama-3" in model_name.lower():
        answer = answer.split("<|start_header_id|>assistant<|end_header_id|>\n\n", 1)[1].split("<|eot_id|>", 1)[0].lstrip()
    elif "ministral" in model_name.lower():
        answer = answer.split("[/INST]", 1)[1].split("</s>", 1)[0].lstrip()
    else:
        answer = answer.lstrip()
    return answer

def generate_outputs(model, tokenizer, dataloader, model_name, output_file_name, use_sampler=False, max_new_tokens=2048):
    output_file = open(f"{output_file_name}", "w", buffering=1)
    generated_batches = []
    dataloader = accelerator.prepare(dataloader)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        params = {
            "max_new_tokens": max_new_tokens,
        }
        
        if use_sampler:
            params.update({
                "repetition_penalty": 1.1,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "temperature": 1.0
            })
            
        with torch.no_grad():
            outputs = model.generate(
                batch['without_answer_input_ids'],
                attention_mask=batch['without_answer_attention_mask'],
                **params,
                pad_token_id=tokenizer.eos_token_id
            )

        # decode the outputs and add
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        generated_batches.extend(decoded_outputs)
        
        for i in range(len(decoded_outputs)):
            pure_answer = postprocess_answer(decoded_outputs[i], batch['prompt'][i], model_name)
            output_file.write(json.dumps(
                {"prompt": batch['prompt'][i], "output": pure_answer,
                    "question": batch['question'][i], "answer": batch['answer'][i]}) + "\n")
  
    output_file.close()


if __name__ == "__main__":
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, padding_side='left', add_bos_token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    
    # Create dataset and dataloader
    if args.dataset == "GSM8K":
        dataset = GSM8KDataset(
            tokenizer=tokenizer,
            model_name=args.model,
            dataset_path="../eval_data/gsm8k/test.jsonl"
        )
    elif args.dataset == "MATH500":
        dataset = MATHDataset(
            tokenizer=tokenizer,
            dataset_path="../eval_data/math500/test.jsonl",
            model_name=args.model,
            data_type="MATH500"
        )
    elif args.dataset == "MATH":
        dataset = MATHDataset(
            tokenizer=tokenizer,
            dataset_path="datasets/MATH/test",
            model_name=args.model,
            data_type="MATH"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Generate outputs
    generate_outputs(
        model,
        tokenizer,
        dataloader,
        model_name=args.model,
        output_file_name=args.output_file_name,
        use_sampler=args.use_sampler,
        max_new_tokens=args.max_new_tokens
    )
