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
arg.add_argument("--dataset", type=str, required=True, choices=["SinglePrompt", "WDMPBio", "WDMPChem", "WDMPCyber"])
args = arg.parse_args()
logger.info(f"Generating outputs on {DEVICE}")
print(args)


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


class WDMPBioDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, model_name, max_length=1024):
        self.tokenizer = tokenizer
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        self.chat_template = get_chat_template(model_name)

        self.prompts = []
        for i in range(len(self.data)):
            prompt = self.data[i]['instruction']
            self.prompts.append(self.chat_template.format(prompt=prompt))
        logger.info(f"Loaded {len(self.prompts)} prompts")

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
        }

class WDMPChemDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, model_name, max_length=1024):
        self.tokenizer = tokenizer
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        self.chat_template = get_chat_template(model_name)

        self.prompts = []
        for i in range(len(self.data)):
            prompt = self.data[i]['instruction']
            self.prompts.append(self.chat_template.format(prompt=prompt))
        logger.info(f"Loaded {len(self.prompts)} prompts")

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
        }

class WDMPCyberDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, model_name, max_length=2048):
        self.tokenizer = tokenizer
        with open(jsonl_path, 'r') as f:
            self.data = [json.loads(line) for line in f]

        self.chat_template = get_chat_template(model_name)

        self.prompts = []
        for i in range(len(self.data)):
            prompt = self.data[i]['instruction']
            self.prompts.append(self.chat_template.format(prompt=prompt))
        logger.info(f"Loaded {len(self.prompts)} prompts")

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
        # Let accelerate and device_map handle device placement automatically
        
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
            try:
                pure_answer = postprocess_answer(decoded_outputs[i], batch['prompt'][i], model_name)
            except Exception as e:
                continue
            if "<think>" in pure_answer and re.search(r'</think\d*>', pure_answer):
                # Find the last occurrence of </think> or </thinkN> pattern
                think_tags = re.findall(r'</think\d*>', pure_answer)
                if think_tags:
                    last_tag = think_tags[-1]
                    ans_wo_cot = pure_answer.split(last_tag)[-1]
                    output_file.write(json.dumps({"prompt": batch['prompt'][i], "output": pure_answer, "ans_wo_cot": ans_wo_cot}) + "\n")
                else:
                    output_file.write(json.dumps({"prompt": batch['prompt'][i], "output": pure_answer}) + "\n")
            else:
                output_file.write(json.dumps({"prompt": batch['prompt'][i], "output": pure_answer}) + "\n")
  
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
    if args.dataset == "WDMPBio":
        dataset = WDMPBioDataset(
            tokenizer=tokenizer,
            jsonl_path="./dataset/wmdpbio-qwen/test.jsonl",
            model_name=args.model
        )
    elif args.dataset == "WDMPChem":
        dataset = WDMPChemDataset(
            tokenizer=tokenizer,
            jsonl_path="./dataset/wmdpchem-qwen/test.jsonl",
            model_name=args.model
        )
    elif args.dataset == "WDMPCyber":
        dataset = WDMPCyberDataset(
            tokenizer=tokenizer,
            jsonl_path="./dataset/wmdpcyber-qwen/test.jsonl",
            model_name=args.model
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
