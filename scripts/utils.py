# https://github.com/pacman100/DHS-LLM-Workshop/blob/main/chat_assistant/training/utils.py
import os
import sys
import json
import jsonlines
import logging
import transformers

from enum import Enum
import os
import torch
import pandas as pd
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from tqdm import tqdm
from peft import LoraConfig
from peft.tuners.lora import LoraLayer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def dump_jsonl(data, output_path, append=False):
    """
    write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')


def get_logger(output_dir=None, description=None):
    if output_dir != None:
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[
                logging.FileHandler(
                    f"{output_dir}/log_{description}.txt", mode="a", delay=False
                ),
                logging.StreamHandler(sys.stdout),
            ],
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
        )

    logger = logging.getLogger('NLI4CT')
    logger.setLevel(10)
    return logger


def get_verification_instructions(raw_data, cot=False):
    keys = list(raw_data.keys())
    instructions = []
    for key in keys:
        sample = raw_data[key]
        statement = sample['Statement']
        label = sample['Label']
        inference_type = sample['Type']
        section = sample['Section_id']
        primary_id = sample['Primary_id']

        with open(f'./data/CT json/{primary_id}.json') as file:
            primary_content = json.load(file)
        primary_section = primary_content[section]
        primary_section = [element.strip() for element in primary_section]
        primary = ' '.join(primary_section)
        if inference_type == 'Comparison':
            secondary_id = sample['Secondary_id']
            with open(f'./data/CT json/{secondary_id}.json') as file:
                secondary_content = json.load(file)
            secondary_section = secondary_content[section]
            secondary_section = [element.strip() for element in secondary_section]
            secondary = ' '.join(secondary_section)
        if cot:
            if inference_type == 'Single':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nStatement: {statement}\nVerify whether the statement is entailed in the primary clinical trial report with Entailment or Contradiction. [/INST]\nLet's take it step by step:"

            elif inference_type == 'Comparison':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nStatement: {statement}\nVerify whether the statement is entailed in the clinical trial reports with Entailment or Contradiction. [/INST]\nLet's take it step by step:"

        else:
            if inference_type == 'Single':
                instruction = f'<s>[INST] Primary clinical trial report:\n{primary}\nVerify whether the following statement is entailed in the primary clinical trial report with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '

            elif inference_type == 'Comparison':
                instruction = f'<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nVerify whether the following statement is entailed in the clinical trial reports with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '

        instructions.append({'text': instruction})

    return instructions


def get_nli4ct_cot_data(data):
    processed_data = []
    for sample in data:
        for key, value in sample.items():
            instruction = value['instruction']
            label = value['label']
            output = value['output']
            cot_with_label = output + f'\n\nVerification: {label} </s>'
            processed_data.append({'key': key, 'instruction': instruction, 'output': cot_with_label})
    return processed_data


def get_verification_only_instructions(raw_data):
    keys = list(raw_data.keys())
    instructions = []
    for key in keys:
        sample = raw_data[key]
        statement = sample['Statement']
        label = sample['Label']
        inference_type = sample['Type']
        section = sample['Section_id']
        primary_id = sample['Primary_id']

        with open(f'./data/CT json/{primary_id}.json') as file:
            primary_content = json.load(file)
        primary_section = primary_content[section]
        primary_section = [element.strip() for element in primary_section]
        primary = ' '.join(primary_section)
        if inference_type == 'Comparison':
            secondary_id = sample['Secondary_id']
            with open(f'./data/CT json/{secondary_id}.json') as file:
                secondary_content = json.load(file)
            secondary_section = secondary_content[section]
            secondary_section = [element.strip() for element in secondary_section]
            secondary = ' '.join(secondary_section)

        if inference_type == 'Single':
            instruction = f'<s>[INST] Primary clinical trial report:\n{primary}\nVerify whether the following statement is entailed in the primary clinical trial report with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '

        elif inference_type == 'Comparison':
            instruction = f'<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nVerify whether the following statement is entailed in the clinical trial reports with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '

        instructions.append({'key': key, 'instruction': instruction, 'output': f'{label} </s>'})

    return instructions


def create_nli4ct_cot_dataset(train_file, dev_file, output_train_file, output_dev_file):
    with open(train_file) as file:
        train = json.load(file)
    with open(dev_file) as file:
        dev = json.load(file)
    processed_train = get_nli4ct_cot_data(train)
    processed_dev = get_nli4ct_cot_data(dev)
    processed_train_data = {"data": processed_train}
    processed_dev_data = {"data": processed_dev}
    with open(output_train_file, 'w') as file:
        json.dump(processed_train_data, file)
    with open(output_dev_file, 'w') as file:
        json.dump(processed_dev_data, file)


def create_nli4ct_label_only_dataset(train_file, dev_file, output_train_file, output_dev_file):
    with open(train_file) as file:
        train = json.load(file)
    with open(dev_file) as file:
        dev = json.load(file)
    processed_train = get_verification_only_instructions(train)
    processed_dev = get_verification_only_instructions(dev)
    # add dev data for training
    # processed_train += processed_dev
    processed_train_data = {"data": processed_train}
    processed_dev_data = {"data": processed_dev}
    with open(output_train_file, 'w') as file:
        json.dump(processed_train_data, file)
    with open(output_dev_file, 'w') as file:
        json.dump(processed_dev_data, file)


if __name__ == '__main__':
    create_nli4ct_label_only_dataset('./data/train.json', './data/dev.json',
                                     './data/fsdp_label_only_train.json',
                                     './data/fsdp_label_only_test.json')
    create_nli4ct_cot_dataset('./data/prompts/gpt4/gpt4_train_cot_verification_selected.json',
                              './data/prompts/gpt4/gpt4_dev_cot_verification_selected.json',
                              './data/fsdp_gpt4_cot_verification_train.json',
                              './data/fsdp_gpt4_cot_verification_test.json')
