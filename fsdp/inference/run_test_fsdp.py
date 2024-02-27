import json
import os
import torch
import jsonlines
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import sleep
from tqdm import tqdm
from scripts.utils import get_logger, dump_jsonl


def get_prompt(mode, inference_type, statement, primary, secondary=None):
    if mode == 'verification_only':
        if inference_type == 'Single':
            prompt = f'<s>[INST] Primary clinical trial report:\n{primary}\nVerify whether the following statement is entailed in the primary clinical trial report with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '
        elif inference_type == 'Comparison':
            prompt = f'<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nVerify whether the following statement is entailed in the clinical trial reports with Entailment or Contradiction.\nStatement: {statement} [/INST]\nVerification: '
    elif mode == 'cot_verification':
        if inference_type == 'Single':
            prompt = f"<s>[INST] Primary clinical trial report:\n{primary}\nStatement: {statement}\nVerify whether the statement is entailed in the primary clinical trial report with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"

        elif inference_type == 'Comparison':
            prompt = f"<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nStatement: {statement}\nVerify whether the statement is entailed in the clinical trial reports with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"
    return prompt


def run(model, tokenizer, mode, model_category, model_size, dataset, appendix, test_file, start_index, end_index,
        sampling=False):
    description = f'{model_category}_{model_size}_{mode}_{dataset}_{appendix}'
    logger = get_logger('./logs/', description)
    output_path = f'./data/prompts/{model_size}/{description}.jsonl'
    with open(test_file) as file:
        data = json.load(file)
    data_keys = list(data.keys())[start_index:end_index]
    if os.path.isfile(output_path):
        generated_data = list(jsonlines.open(output_path))
        generated_keys = [list(element.keys())[0] for element in generated_data]
    else:
        generated_keys = []
    not_finished_keys = [key for key in data_keys if key not in generated_keys]

    for key in tqdm(not_finished_keys):
        sample = data[key]
        statement = sample['Statement']
        inference_type = sample['Type']
        section = sample['Section_id']
        primary_id = sample['Primary_id']
        with open(f'./data/CT json/{primary_id}.json') as file:
            primary_content = json.load(file)
        primary_section = primary_content[section]
        primary_section = [element.strip() for element in primary_section]
        primary = ' '.join(primary_section)
        secondary = None

        if inference_type == 'Comparison':
            secondary_id = sample['Secondary_id']
            with open(f'./data/CT json/{secondary_id}.json') as file:
                secondary_content = json.load(file)
            secondary_section = secondary_content[section]
            secondary_section = [element.strip() for element in secondary_section]
            secondary = ' '.join(secondary_section)

        prompt = get_prompt(mode=mode, inference_type=inference_type, statement=statement, primary=primary,
                            secondary=secondary, )

        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
        logger.info(inputs.shape[1])
        # if inputs.shape[1] < 2048 or inputs.shape[1] > 2500:
        if inputs.shape[1] > 4000:
            pass
        else:
            if sampling:
                generated_ids = model.generate(input_ids=inputs, max_new_tokens=500, do_sample=True, top_k=50,
                                               temperature=0.7, num_return_sequences=10)
                output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)

            else:
                generate_ids = model.generate(input_ids=inputs, max_new_tokens=900, do_sample=False)
                output = \
                    tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                        0]
                logger.info(output)
            dump_jsonl([{key: {'prompt': prompt, 'info': sample, 'output': output}}], output_path, append=True)

            logger.info("\n==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    args = parser.parse_args()
    data_info = {"dataset": "test", "test_file": "./data/test.json"}
    LORA_WEIGHTS = f'./checkpoints/cot'
    BASE_MODEL = '/path_to_pretrained_llm/mixtral/Mixtral-8x7B-Instruct-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.eos_token_id = 2
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map='balanced',
                                                 )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.bfloat16, device_map='balanced',
                                      )

    start_index = args.start_index
    end_index = args.end_index
    run(model=model, tokenizer=tokenizer, mode='selfcon_cot', model_category='mixtral_lora',
        model_size='7B', dataset=data_info['dataset'], test_file=data_info['test_file'], start_index=start_index,
        end_index=end_index, appendix=f'fsdp_5epoch_w_dev_{start_index}_{end_index}', sampling=True)
