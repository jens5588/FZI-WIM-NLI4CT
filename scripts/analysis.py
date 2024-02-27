import collections
import json
import jsonlines
from openai import OpenAI
import random
import string
import torch
import pandas as pd
from time import sleep
from fairseq.data.data_utils import collate_tokens
from sklearn.metrics import classification_report
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import dump_jsonl, get_logger


def extract_reasoning_output(input_path):
    data = list(jsonlines.open(input_path))
    labels = []
    predictions = []
    errors = []
    for idx, element in enumerate(data):
        label = element['info']['Label']
        output = element['output'].replace('\n', ' ')
        processed_output = output.strip().translate(str.maketrans('', '', string.punctuation))
        processed_output = processed_output.split(' ')

        predicted_label = None
        if len(set([word.lower() for word in processed_output[:10]]).intersection(
                {'false', 'incorrect', 'no', 'not', 'wrong', 'contradictions'})) > 0:
            predicted_label = 'Contradiction'
        elif len(set([word.lower() for word in processed_output[:10]]).intersection(
                {'true', 'correct', 'yes', 'right', 'entailments'})) > 0:
            predicted_label = 'Entailment'
        else:
            print(output)
            errors.append(idx)
        labels.append(label)
        predictions.append(predicted_label)
    to_evaluated_labels = [labels[i] for i in range(len(data)) if i not in errors]
    to_evaluated_predictions = [predictions[i] for i in range(len(data)) if i not in errors]
    assert len(to_evaluated_labels) == len(to_evaluated_predictions)
    print(f'not evaluated: {errors}')
    print(f'{len(to_evaluated_labels)}/{len(data)}')
    print(classification_report(to_evaluated_labels, to_evaluated_predictions))


def evaluate_verification_result(input_file):
    data = list(jsonlines.open(input_file))
    gold_labels = []
    predicted_labels = []
    for idx, sample in enumerate(data):
        for key, value in sample.items():
            prompt = value['prompt']
            info = value['info']
            output = value['output']
            gold_labels.append(info['Label'])
            predicted_label = output[len(prompt) - 3:].strip()
            if predicted_label in ['Entailment', 'Contradiction']:
                predicted_labels.append(predicted_label)
            else:
                print(predicted_label)
                print(f'Error in generation with idx {idx}')
                predicted_labels.append(None)
    print(classification_report(gold_labels, predicted_labels, digits=4))


def get_cot_verification_instructions(raw_data, given_label=False):
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
        if given_label:
            if inference_type == 'Single':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nStatement: {statement}\nVerify whether the statement is entailed in the primary clinical trial report with Entailment or Contradiction.\nVerification: {label} [/INST]\nLet's justify the verification step by step:"
            elif inference_type == 'Comparison':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nStatement: {statement}\nVerify whether the statement is entailed in the clinical trial reports with Entailment or Contradiction.\nVerification: {label}  [/INST]\nLet's justify the verification step by step:"

        else:
            if inference_type == 'Single':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nStatement: {statement}\nVerify whether the statement is entailed in the primary clinical trial report with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"

            elif inference_type == 'Comparison':
                instruction = f"<s>[INST] Primary clinical trial report:\n{primary}\nSecondary clinical trial report:\n{secondary}\nStatement: {statement}\nVerify whether the statement is entailed in the clinical trial reports with Entailment or Contradiction. [/INST]\nLet's verify it step by step:"
        instructions.append({key: {'instruction': instruction, 'label': label, 'statement': statement}})

    return instructions


def cot_verification_gpt4(raw_data, output_file, model='gpt-4', temperature=0, max_tokens=400):
    logger = get_logger('./logs/', 'cot_train_part2')
    data = get_cot_verification_instructions(raw_data)
    client = OpenAI(
        api_key="own_key",
    )
    error_idx = []
    for idx, element in enumerate(data):
        logger.info(f'Current index: {idx}')
        for key, value in element.items():
            instruction = value['instruction']
            label = value['label']
            statement = value['statement']
            logger.info(f'Instruction: {instruction}')
            logger.info(f'Label: {label}')
            message = [
                {"role": "user", "content": instruction},
            ]
            try:
                response = client.chat.completions.create(model=model, messages=message, temperature=temperature,
                                                          max_tokens=max_tokens)
                generated_text = response.choices[0].message.content
                logger.info(generated_text)
                dump_jsonl(
                    [{key: {'instruction': instruction, 'statement': statement, 'label': label,
                            'output': generated_text}}],
                    output_file, append=True)
            except Exception as e:
                logger.info(e)
                error_idx.append(idx)
        sleep(15)
    logger.info(f'Error index: {error_idx}')


def bart_nli_label(nli_model, premise_hypothesis_pairs):
    softmax = torch.nn.Softmax(dim=1)
    sentence_pair_tokens = [nli_model.encode(pair[0], pair[1]) for pair in premise_hypothesis_pairs]
    batch = collate_tokens(sentence_pair_tokens, pad_idx=1)
    with torch.no_grad():
        logprobs = nli_model.predict('mnli', batch)
    labels = softmax(logprobs).argmax(dim=1).tolist()
    # print(labels)
    labels = [int(label) for label in labels]
    if 2 in labels:
        return 1
    else:
        return 0


def check_conclusion_label(nli_model, conclusion):
    entail_claims = ['the statement is entailed', 'the statement is entailed in the clinical report',
                     'the statement is entailed in the clinical reports',
                     'the statement is entailment',
                     'the statement is consistent with the information provided in the report',
                     'the statement is supported']
    contradict_claims = ['contradiction', 'the statement is contradicted', 'the statement is a contradiction',
                         'the statement contradicts the clinical reports',
                         'the statement is not entailed in the clinical reports',
                         'the statement is not entirely entailed in the reports',
                         'the statement is not entirely correct',
                         'it"s a contradiction', 'the correct answer is contradiction',
                         'the answer is contradiction',
                         'the statement is not entirely correct']

    entailment_pairs = zip([conclusion] * len(entail_claims), entail_claims)
    contradict_pairs = zip([conclusion] * len(contradict_claims), contradict_claims)

    entailment_label = bart_nli_label(nli_model, entailment_pairs)
    contradict_label = bart_nli_label(nli_model, contradict_pairs)

    if entailment_label + contradict_label == 1:
        if entailment_label == 1:
            return 'Entailment'
        elif contradict_label == 1:
            return 'Contradiction'
    elif entailment_label + contradict_label == 2:
        return 'both_true'

    else:
        return 'no label can be assigned'


def apply_rule(conclusion):
    predicted_label = None
    if ('contradiction' in conclusion and 'entail' not in conclusion):
        predicted_label = 'Contradiction'

    elif 'entail' in conclusion and 'not entail' not in conclusion and 'contradict' not in conclusion and 'partial' not in conclusion:
        predicted_label = 'Entailment'

    return predicted_label


def extract_gpt4_cot_label(input_file, output_file):
    data = list(jsonlines.open(input_file))
    # load nli model
    nli_model = torch.hub.load('pytorch/fairseq', 'bart.large.mnli')
    nli_model.eval()
    device = 'cuda'
    nli_model.to(device)
    correct = 0
    label_assigned = 0
    correct_samples = []
    for idx, sample in enumerate(data):
        print(idx)
        for key, value in sample.items():
            output = value['output'].lower()
            label = value['label']
            if 'therefore,' in output:
                conclusion = output.split('therefore,')[-1]
            elif 'so,' in output:
                conclusion = output.split('so,')[-1]
            elif '\n\n' in output:
                conclusion = output.split('\n\n')[-1]
            else:
                conclusion = output.split('\n')[-1]

            if '\n' in conclusion:
                conclusion = conclusion.split('\n')[-1]
            nli_label = check_conclusion_label(nli_model, conclusion)

            if nli_label in ['Entailment', 'Contradiction']:
                label_assigned += 1
                if nli_label == label:
                    correct += 1
                    correct_samples.append(sample)
            # nli model has both entailment and contradiction predicted
            elif nli_label == 'both_true':
                predicted_label = apply_rule(conclusion)
                if predicted_label is not None:
                    label_assigned += 1
                    if predicted_label == label:
                        correct += 1
                        correct_samples.append(sample)

            # nli model has predicted neither entailment nor contradiction
            else:
                predicted_label = apply_rule(conclusion)
                if predicted_label == label:
                    correct_samples.append(sample)
                    correct += 1
                    correct_samples.append(sample)

                else:
                    print(f'Label: {label}')
                    print(conclusion)
                    # 104, 765, 1284
    with open(output_file, 'w') as file:
        json.dump(correct_samples, file)
    print(f'Label assigned {label_assigned}')
    print(f'Correct label {correct}')


def check_gpt4_label(input_file_selected, input_file_raw=None):
    with open(input_file_selected) as file:
        data = json.load(file)['data']
    with open(input_file_raw) as file:
        train = json.load(file)
    for element in random.sample(data, 100):
        key = element['key']
        print(f'Label {train[key]["Label"]}')
        output = element['output'].split('\n\n')[-2:]
        print(output)
        print('######################')


def combine_generation(output_file):
    with open('./data/test.json') as file:
        test_data = json.load(file)
    test_keys = list(test_data.keys())
    input_data1 = list(
        jsonlines.open(
            './data/prompts/7B/mixtral_lora_7B_cot_verification_test_fsdp_5epoch_w_dev_0_1800.jsonl'))

    input_data2 = list(
        jsonlines.open(
            './data/prompts/7B/mixtral_lora_7B_cot_verification_test_fsdp_5epoch_w_dev_1800_3600.jsonl'))

    input_data3 = list(
        jsonlines.open(
            './data/prompts/7B/mixtral_lora_7B_cot_verification_test_fsdp_5epoch_w_dev_3600_5500.jsonl'))
    cot_data = input_data1 + input_data2 + input_data3

    print(f'Total generation: {len(cot_data)}')
    # verified = 0
    deduplicated = []
    deduplicated_keys = []
    for element in cot_data:
        for key, value in element.items():
            if key not in deduplicated_keys:
                deduplicated.append(element)
                deduplicated_keys.append(key)
                dump_jsonl([element], output_file, append=True)
    print(len(deduplicated_keys))
    assert len(deduplicated) == len(deduplicated_keys) == len(cot_data)

    '''cot_data_keys = [list(element.keys())[0] for element in cot_data]
    all_samples_generated = [1 for key in test_keys if key in cot_data_keys]
    print(sum(all_samples_generated))
    counter = collections.Counter(cot_data_keys)
    d = dict(counter)
    duplicates = []
    for key, value in d.items():
        if value == 2:
            duplicates.append(key)
    duplicate_output = []
    for key in duplicates:
        for element in cot_data:
            for e_key, value in element.items():
                if e_key == key:
                    prompt = value['prompt']
                    duplicate_output.append(value['output'][len(prompt) - 2:])
    for element in duplicate_output:
        print(element)
        print('==================')'''


def extract_label(generation_file):
    with open('./data/test.json') as file:
        test_data = json.load(file)
    test_keys = list(test_data.keys())
    generated_data = list(jsonlines.open(generation_file))
    verification = {}
    unverified_keys = []
    verified_keys = []
    for element in generated_data:
        for key, value in element.items():
            prompt = value['prompt']
            output = value['output']
            predicted_label = output.split('Verification: ')[-1].strip()
            if predicted_label in ['Entailment', 'Contradiction']:
                verification[key] = {"Prediction": predicted_label}
                verified_keys.append(key)
            elif predicted_label.lower() in ['not enough information', 'inconclusive', 'undetermined', 'not entailed',
                                             'neutral', 'undefined', 'indeterminate']:
                verification[key] = {"Prediction": 'Contradiction'}
                verified_keys.append(key)
            else:
                unverified_keys.append(key)
                print(key)
                print(output[len(prompt) - 2:])
                print('===================')

    generated_keys = verified_keys + unverified_keys
    assert len(set(test_keys).intersection(generated_keys)) == len(test_keys)
    print(len(verification))
    print(len(unverified_keys))
    verification['21ae5802-04f7-4f06-86d6-9ee8ee9cdd84'] = {'Prediction': 'Contradiction'}
    verification['2a13f623-4957-4e7e-9968-03f64bb6f5f3'] = {'Prediction': 'Contradiction'}
    verification['12746144-95c9-4656-abdf-8a49f78c7049'] = {'Prediction': 'Contradiction'}
    verification['fbd6b437-ba96-46fd-933f-8299e0efb0e5'] = {'Prediction': 'Entailment'}
    verification['db7671cb-a230-402e-beda-738af0753332'] = {'Prediction': 'Contradiction'}
    verification['aa861098-a5f8-473e-be7d-eba94ccb4ab1'] = {'Prediction': 'Contradiction'}
    verification['7814620b-23de-463b-8fd1-efdf51ce26b3'] = {'Prediction': 'Contradiction'}
    verification['7714e6f7-814a-4088-9116-60f5ceef4336'] = {'Prediction': 'Contradiction'}
    verification['44b41322-9fb1-4a02-b2c5-4faead5489aa'] = {'Prediction': 'Contradiction'}
    verification['8fcaece4-a0c7-44bf-82b0-50e494825b81'] = {'Prediction': 'Contradiction'}
    verification['47b36e1f-90d5-4a42-b359-95783c7ae965'] = {'Prediction': 'Contradiction'}
    verification['f82d082e-896d-4d9e-ad92-d64b655dafb8'] = {'Prediction': 'Contradiction'}

    '''verification['6ee2e04c-88fc-4447-a477-18b2564aca4e'] = {'Prediction': 'Contradiction'}
    verification['47daf120-3939-4e32-bba5-d86ce54a9cbd'] = {'Prediction': 'Contradiction'}
    verification['32ce7a8e-20c4-42d9-a4d4-e3b5d580331e'] = {'Prediction': 'Entailment'}
    verification['dc867a3f-6005-49f7-bcd8-1c7db27b00c1'] = {'Prediction': 'Contradiction'}
    verification['88a40578-17e1-499b-8b7b-10a85e2c3d88'] = {'Prediction': 'Contradiction'}
    verification['b7b61d5d-629d-43a4-8d75-5897864b6443'] = {'Prediction': 'Entailment'}
    verification['6fbe26db-3420-4617-9f84-1663aa9be297'] = {'Prediction': 'Contradiction'}
    verification['33472836-f751-4de3-8b21-92e2d9e3d7ed'] = {'Prediction': 'Contradiction'}'''
    verification_keys = list(verification.keys())
    assert len(set(test_keys).intersection(verification_keys)) == len(test_keys)
    with open('./data/results.json', 'w') as file:
        json.dump(verification, file)


def most_common(lst):
    return max(set(lst), key=lst.count)


def check_decided(output):
    decided = False
    predicted_label_major = None
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
        if other_label in d and d[predicted_label_major] - d[other_label] > 0:
            decided = True
        elif other_label not in d:
            decided = True
        else:
            print(d)

    return decided, predicted_label_major


def extract_selfcon_final_label(tokenizer, input_file):
    with open('./data/test.json') as file:
        test_data = json.load(file)
    num_decided = 0
    undecided_length = []
    duplicates = []
    test_keys = list(test_data.keys())
    verification = {}
    generation = list(jsonlines.open(input_file))
    print(len(generation))
    for idx, element in enumerate(generation):
        for key, value in element.items():
            prompt = value['prompt']
            output = value['output']
            if len(list(set(output))) != len(output):
                duplicates.append([key, len(list(set(output))), len(output)])
            decided, predicted_label_major = check_decided(output)
            if decided:
                num_decided += 1
                verification[key] = {'Prediction': predicted_label_major}
            else:
                prompt_length = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
                undecided_length.append(prompt_length)
                print(f'{key}--{len(output)}--{prompt_length}')
                random.seed(123)
                predicted_label = random.choice(['Entailment', 'Contradiction'])
                verification[key] = {'Prediction': predicted_label}

                '''for p in output:
                    print(p[len(prompt) - 2:])
                    print('=================================')'''
                print('\n\n')
    print(num_decided)
    print(sorted(undecided_length))
    print(len(duplicates))

    verification['6ee2e04c-88fc-4447-a477-18b2564aca4e'] = {'Prediction': 'Contradiction'}
    verified_keys = list(verification.keys())

    assert len(list(set(verified_keys).intersection(set(test_keys)))) == len(test_keys)

    with open('./data/results.json', 'w') as file:
        json.dump(verification, file)


def extract_cot_selfcon_label(tokenizer, input_file):
    with open('./data/test.json') as file:
        test_data = json.load(file)
    test_keys = list(test_data.keys())
    generation = list(jsonlines.open(input_file))
    print(len(generation))
    not_finished_length = []
    verification = {}
    all = 0
    major = 0
    neutral = 0
    decided = 0
    tied = 0
    for idx, element in enumerate(generation):
        predicted_labels = []
        for key, value in element.items():
            prompt = value['prompt']
            output = value['output']
            for part in output:
                predicted_label = part.split('Verification: ')[-1].strip()
                if predicted_label in ['Entailment', 'Contradiction']:
                    predicted_labels.append(predicted_label)
                elif predicted_label.lower() in ['not entailed']:
                    predicted_labels.append('Contradiction')
                elif predicted_label.lower() in ['not enough information', 'inconclusive', 'undetermined',
                                                 'neutral', 'undefined', 'indeterminate']:
                    predicted_labels.append('Neutral')
                else:
                    pass
        if len(predicted_labels) == 10:
            all += 1
        if len(predicted_labels) > 2:
            major += 1
            prompt_length = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            not_finished_length.append(prompt_length)
            counter = collections.Counter(predicted_labels)
            d = dict(counter)
            predicted_label_major = most_common(predicted_labels)
            labels = ['Entailment', 'Contradiction']
            labels.pop(labels.index(predicted_label_major))
            other_label = labels[0]
            if other_label in d and d[predicted_label_major] - d[other_label] == 0:
                print(d)
                print('###################')
                tied += 1
            if other_label in d and d[predicted_label_major] - d[other_label] > 0:
                decided += 1
            elif other_label not in d:
                decided += 1
            verification[key] = {'Prediction': predicted_label_major}
            if predicted_label_major == 'Neutral':
                neutral += 1
                verification[key] = {'Prediction': 'Contradiction'}

        else:
            # print(key)
            prompt_length = tokenizer(prompt, return_tensors="pt").input_ids.shape[1]
            not_finished_length.append(prompt_length)

            # print(idx)
            '''for part in output:
                print(part[len(prompt) - 2:])
                print('=============================')'''
    '''verification['1672004b-d668-4441-95b4-eaab6fd70818'] = {'Prediction': 'Contradiction'}
    verification['04901c04-5a0e-46cb-8684-96ff864921fe'] = {'Prediction': 'Contradiction'}
    verified_keys = list(verification.keys())
    print(len(list(set(verified_keys))))
    assert len(list(set(verified_keys).intersection(set(test_keys)))) == len(test_keys)'''
    print(all)
    print(major)
    print(neutral)
    print(decided)
    print(tied)
    # s = pd.Series(not_finished_length)
    # print(s.describe([0.5, 0.8, 0.9, 0.95]))
    # with open('./data/prompts/7B/results.json', 'w') as file:
    #    json.dump(verification, file)


def extract_selfcon_label(output):
    predicted_label_major = None
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
    if len(predicted_labels) > 2:
        predicted_label_major = most_common(predicted_labels)

    return predicted_label_major


def compare_cot_selfcon(selfcon_file, cot_file):
    selfcon_data = list(jsonlines.open(selfcon_file))
    cot_data = list(jsonlines.open(cot_file))
    cot_data_keys = [list(element.keys())[0] for element in cot_data]
    for idx, element in enumerate(selfcon_data):
        for key, value in element.items():
            label = value['info']['Label']
            output = value['output']
            selfcon_label = extract_selfcon_label(output)
            cot_index = cot_data_keys.index(key)
            for c_key, c_value in cot_data[cot_index].items():
                cot_output = c_value['output']
                cot_label = cot_output.split('Verification: ')[-1].strip()
            if selfcon_label == label and cot_label != label:
                print(idx)
                print(key)


def compare_cot_selfcon_test(selfcon, cot, selfcon_gen, cot_gen, tokenizer):
    different_predictions = []
    with open(selfcon) as file:
        selfcon_data = json.load(file)
    with open(cot) as file:
        cot_data = json.load(file)
    selfcon_gen_data = list(jsonlines.open(selfcon_gen))
    cot_gen_data = list(jsonlines.open(cot_gen))
    selfcon_keys = list(selfcon_data.keys())
    for key in selfcon_keys:
        if selfcon_data[key]['Prediction'] != cot_data[key]['Prediction']:
            different_predictions.append(key)
    selfcon_gen_keys = [list(element.keys())[0] for element in selfcon_gen_data]
    cot_gen_keys = [list(element.keys())[0] for element in cot_gen_data]
    for key in different_predictions:
        cot_index = cot_gen_keys.index(key)
        selfcon_index = selfcon_gen_keys.index(key)
        for c_key, c_value in cot_gen_data[cot_index].items():
            prompt = c_value['prompt']
            c_output = c_value['output'][len(prompt) - 2:]
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            if 150 < inputs.shape[1] < 200:
                print(cot_index)
                print(prompt)
                print(c_output)
                for s_key, s_value in selfcon_gen_data[selfcon_index].items():
                    s_output = s_value['output']
                    for p in s_output:
                        print(p[len(prompt) - 2:])
                        print('===================')
                print('##########################')
                print('\n\n')


def check_token_length(tokenizer):
    num_tokens = []
    selfcon_gen_data = list(
        jsonlines.open('./data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    for element in selfcon_gen_data:
        for key, value in element.items():
            prompt = value['prompt']
            inputs = tokenizer(prompt, return_tensors="pt").input_ids
            num_tokens.append(inputs.shape[1])
    s = pd.Series(num_tokens)
    print(s.describe())


def extract_label_only_label(generation_file):
    with open('./data/test.json') as file:
        test_data = json.load(file)
    test_keys = list(test_data.keys())
    generated_data = list(jsonlines.open(generation_file))
    verification = {}
    unverified_keys = []
    verified_keys = []
    for element in generated_data:
        for key, value in element.items():
            prompt = value['prompt']
            output = value['output']
            predicted_label = output.split('Verification: ')[-1].strip()
            if predicted_label in ['Entailment', 'Contradiction']:
                verification[key] = {"Prediction": predicted_label}
                verified_keys.append(key)
            elif predicted_label.lower() in ['not enough information', 'inconclusive', 'undetermined', 'not entailed',
                                             'neutral', 'undefined', 'indeterminate']:
                verification[key] = {"Prediction": 'Contradiction'}
                verified_keys.append(key)
            else:
                unverified_keys.append(key)
                print(key)
                print(output[len(prompt) - 2:])
                print('===================')

    generated_keys = verified_keys + unverified_keys
    assert len(set(test_keys).intersection(generated_keys)) == len(test_keys)
    print(len(verification))
    print(len(unverified_keys))

    with open('./data/results_label_only.json', 'w') as file:
        json.dump(verification, file)


if __name__ == '__main__':
    extract_label_only_label(
        './data/prompts/7B/mixtral_lora_7B_label_only_test_fsdp_5epoch_w_dev.jsonl')
