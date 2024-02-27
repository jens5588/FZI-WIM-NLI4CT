import collections
import json
import os
import torch
import transformers
import jsonlines
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import sleep
from tqdm import tqdm
from scripts.utils import get_logger, dump_jsonl


def most_common(lst):
    return max(set(lst), key=lst.count)


def check_decided(output):
    decided = False
    predicted_labels = []
    for part in output:
        predicted_label = part.split('Verification: ')[-1].strip()
        if predicted_label in ['Entailment', 'Contradiction']:
            predicted_labels.append(predicted_label)
        elif predicted_label.lower() in ['not entailed']:
            predicted_labels.append('Contradiction')
        elif predicted_label.lower() in ['not enough information', 'inconclusive', 'undetermined',
                                         'neutral', 'undefined', 'indeterminate']:
            predicted_labels.append('Neutral')
    if len(predicted_labels) > 8:
        counter = collections.Counter(predicted_labels)
        d = dict(counter)
        predicted_label_major = most_common(predicted_labels)
        labels = ['Entailment', 'Contradiction']
        labels.pop(labels.index(predicted_label_major))
        other_label = labels[0]
        if other_label in d and d[predicted_label_major] - d[other_label] > 1:
            decided = True
        elif other_label not in d:
            decided = True

    return decided


transformers.set_seed(456)


def get_prompt(mode, inference_type, statement, primary, secondary=None):
    if mode == 'verification':
        if inference_type == 'Single':
            prompt = f'<s>[INST] Primary clinical trial report:\n{primary}\nVerify whether the following statement is entailed in the primary clinical trial report with Entailment or Contradiction.\nStatement: {statement} [/Instruct]\nVerification:'
        elif inference_type == 'Comparison':
            prompt = f'<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nVerify whether the following statement is entailed in the clinical trial reports with Entailment or Contradiction.\nStatement: {statement} [/Instruct]\nVerification:'
    elif mode == 'cot_verification':
        if inference_type == 'Single':
            prompt = f"<s>[INST] Primary clinical trial report:\n{primary}\nStatement: {statement}\nVerify whether the statement is entailed in the primary clinical trial report with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"

        elif inference_type == 'Comparison':
            prompt = f"<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nStatement: {statement}\nVerify whether the statement is entailed in the clinical trial reports with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"
    return prompt


# generation_sanity_check is used to check the situation where the verification between Entailment and Contradiction is
# tied. If tied further reasoning chains are generated with different seeds (transformers.set_seed())
def generation_sanity_check(model, tokenizer, mode, model_category, model_size, dataset, appendix, test_file,
                            start_index, end_index):
    description = f'{model_category}_{model_size}_{mode}_{dataset}_{appendix}'
    logger = get_logger('./logs/', description)
    output_path = f'./data/prompts/{model_size}/{description}.jsonl'
    with open(test_file) as file:
        data = json.load(file)
    generated_data = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    for element in tqdm(generated_data[start_index:end_index]):
        for key, value in element.items():
            prompt = value['prompt']
            info = value['info']
            output = value['output']
            decided = check_decided(list(set(output)))
            if not decided:
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
                logger.info(inputs.shape[1])
                if inputs.shape[1] < 2048:
                    num_gen = 25 - len(list(set(output)))
                    if num_gen > 0:
                        generated_ids = model.generate(input_ids=inputs, max_new_tokens=500, do_sample=True, top_k=50,
                                                       temperature=0.7, num_return_sequences=num_gen)
                        output_new = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                        output.extend(output_new)
                        dump_jsonl([{key: {'prompt': prompt, 'info': info, 'output': output}}], output_path,
                                   append=True)
                    else:
                        dump_jsonl([element], output_path, append=True)
                else:
                    dump_jsonl([element], output_path, append=True)
            else:
                dump_jsonl([element], output_path, append=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)
    args = parser.parse_args()
    data_info = {"dataset": "test", "test_file": "./data/test.json"}
    # max_memory setup for 8*A100-40GB
    max_memory = {0: '12GB', 1: '12GB', 2: '12GB', 3: '12GB', 4: '12GB', 5: '12GB', 6: '20GB', 7: '40GB'}

    LORA_WEIGHTS = f'./checkpoints/cot/'
    BASE_MODEL = '/path_to_pretrained_llm/mixtral/Mixtral-8x7B-Instruct-v0.1'
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.eos_token_id = 2
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map='sequential',
                                                 max_memory=max_memory, )
    model = PeftModel.from_pretrained(model, LORA_WEIGHTS, torch_dtype=torch.bfloat16, device_map='sequential',
                                      max_memory=max_memory, )

    start_index = args.start_index
    end_index = args.end_index
    generation_sanity_check(model=model, tokenizer=tokenizer, mode='cot_verification', model_category='mixtral_lora',
                            model_size='7B', dataset=data_info['dataset'], test_file=data_info['test_file'],
                            appendix=f'selfcon_cot_test_fsdp_5epoch_w_dev_sanity_{start_index}_{end_index}',
                            start_index=start_index, end_index=end_index)
