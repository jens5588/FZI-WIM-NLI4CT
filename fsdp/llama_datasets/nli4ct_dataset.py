# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
from datasets import load_dataset


# cot data path: 'train': './data/prompts/gpt4/fsdp_gpt4_cot_verification_combined_train.json',
#                'test': './data/prompts/gpt4/fsdp_gpt4_cot_verification_test.json'
# verification only path:'train': './data/fsdp_label_only_combined_train.json',
#                         'test': './data/fsdp_label_only_test.json'
def get_preprocessed_nli4ct(dataset_config, tokenizer, split):
    dataset = load_dataset("json",
                           data_files={
                               'train': './data/fsdp_gpt4_cot_verification_combined_train.json',
                               'test': './data/fsdp_gpt4_cot_verification_test.json'},
                           field='data', split=split)

    def apply_prompt_template(sample):
        return {
            "prompt": sample['instruction'],
            "output": sample["output"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(sample["prompt"], add_special_tokens=False)
        output = tokenizer.encode(sample["output"], add_special_tokens=False)

        sample = {
            "input_ids": prompt + output,
            "attention_mask": [1] * (len(prompt) + len(output)),
            "labels": [-100] * len(prompt) + output,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
