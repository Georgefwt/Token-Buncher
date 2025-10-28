import re
import os
from datasets import Dataset, load_dataset, concatenate_datasets
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")


def make_prefix(dp, model_type):
    prompt = dp['prompt']
    response = dp['response']
    if model_type == 'base':
        """This works for any base model"""
        prefix = f"### Question:\n {prompt}\n\n### Answer:\n"
    elif model_type == 'ministral':
        prefix = f"<s>[INST]{prompt}[/INST]"
    elif model_type == 'qwen':
        """This works for Qwen Instruct Models"""
        prefix = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif model_type == 'llama-2':
        prefix = f"<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{prompt} [/INST] "
    elif model_type == 'llama-3':
        """This works for Llama 3.x models"""
        prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=10240)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base', help='base, ministral, qwen')

    args = parser.parse_args()

    data_source = 'BeaverTails'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    # check "is_safe" key and filter out true ones
    print("before filter", len(raw_dataset))

    # Filter out examples whose tokenized question length exceeds the threshold.
    MAX_TOKENS = 320
    raw_dataset = raw_dataset.filter(lambda x: len(tokenizer(make_prefix(x, args.template_type)).input_ids) <= MAX_TOKENS)
    print("after length filter", len(raw_dataset))

    FILTER_TYPE = 'unsafe' # select from ['unsafe', 'safe', 'all']
    if FILTER_TYPE == 'unsafe':
        raw_dataset = raw_dataset.filter(lambda x: x['is_safe'] == False)
        train_dataset = raw_dataset.select(range(TRAIN_SIZE))
        test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
        print("train size", len(train_dataset), "test size", len(test_dataset))

    elif FILTER_TYPE == 'safe':
        raw_dataset = raw_dataset.filter(lambda x: x['is_safe'] == True)
        train_dataset = raw_dataset.select(range(TRAIN_SIZE))
        test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
        print("train size", len(train_dataset), "test size", len(test_dataset))

    elif FILTER_TYPE == 'all':
        raw_dataset_unsafe = raw_dataset.filter(lambda x: x['is_safe'] == False)
        print("unsafe after filter", len(raw_dataset_unsafe))
        assert len(raw_dataset_unsafe) > (TRAIN_SIZE + TEST_SIZE)/2

        raw_dataset_safe = raw_dataset.filter(lambda x: x['is_safe'] == True)
        print("safe after filter", len(raw_dataset_safe))
        assert len(raw_dataset_safe) > (TRAIN_SIZE + TEST_SIZE)/2

        # random select TRAIN_SIZE/2 from unsafe and TRAIN_SIZE/2 from safe
        train_dataset_unsafe = raw_dataset_unsafe.select(range(TRAIN_SIZE//2))
        train_dataset_safe = raw_dataset_safe.select(range(TRAIN_SIZE//2))
        train_dataset = concatenate_datasets([train_dataset_unsafe, train_dataset_safe])
        train_dataset = train_dataset.shuffle(seed=42)

        test_dataset_unsafe = raw_dataset_unsafe.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE//2))
        test_dataset_safe = raw_dataset_safe.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE//2))
        test_dataset = concatenate_datasets([test_dataset_unsafe, test_dataset_safe])
        test_dataset = test_dataset.shuffle(seed=42)
        print("train size", len(train_dataset), "test size", len(test_dataset))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, args.template_type)
            solution = {
                "response": example['response']
            }
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "nlp",
                "reward_model": {
                    "style": "model",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
