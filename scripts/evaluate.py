#!/usr/bin/env python3

import json
import jsonlines
import os
import os.path
import sys
import warnings
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.simplefilter('ignore')


def extract_control_set(predictions, gold):
    # check num_control_set
    num_control = 0
    control_predicitons = {}
    for key in gold.keys():
        if "Causal_type" not in gold[key].keys():
            num_control += 1
            control_predicitons[key] = predictions[key]
    print(f'num_control: {num_control}')
    return control_predicitons


def extract_by_intervention(predictions, gold):
    para_predictions = {}
    cont_predictions = {}
    numerical_para_predictions = {}
    numerical_cont_predictions = {}
    definitions_predictions = {}
    for key in predictions.keys():
        if "Intervention" not in gold[key].keys():
            continue
        if gold[key]["Intervention"] == "Paraphrase":
            para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Contradiction":
            cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_paraphrase":
            numerical_para_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Numerical_contradiction":
            numerical_cont_predictions[key] = predictions[key]
        elif gold[key]["Intervention"] == "Text_appended":
            definitions_predictions[key] = predictions[key]

    return para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions


def extract_by_causal_type(predictions, gold):
    predictions_preserving = {}
    predictions_altering = {}
    for key in predictions.keys():
        if "Causal_type" not in gold[key].keys():
            continue
        if gold[key]["Causal_type"][0] == "Preserving":
            predictions_preserving[key] = predictions[key]
        elif gold[key]["Causal_type"][0] == "Altering":
            predictions_altering[key] = predictions[key]
    return predictions_preserving, predictions_altering


def faithfulness(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] != gold[gold[key]["Causal_type"][1]]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Faithfulness = sum(results) / N
    return Faithfulness


def consistency(predictions, gold):
    uuid_list = list(predictions.keys())
    N = len(uuid_list)
    results = []
    for key in uuid_list:
        if predictions[key]["Prediction"] == gold[key]["Label"]:
            results.append(1)
        else:
            results.append(0)
    Consistency = sum(results) / N
    return Consistency


def extract_contrast_set(predictions, gold):
    contrast_predicitons = {}
    for key in predictions.keys():
        if "Causal_type" in gold[key].keys():
            contrast_predicitons[key] = predictions[key]
    return contrast_predicitons


def F1_Recall_Precision(predictions, gold):
    pred_labels = []
    gold_labels = []
    for key in predictions.keys():
        if predictions[key]["Prediction"] == "Entailment":
            pred_labels.append(1)
        else:
            pred_labels.append(0)
        if gold[key]["Label"] == "Entailment":
            gold_labels.append(1)
        else:
            gold_labels.append(0)
    F1 = f1_score(gold_labels, pred_labels)
    Recall = recall_score(gold_labels, pred_labels)
    Precision = precision_score(gold_labels, pred_labels)

    return F1, Recall, Precision


def get_preserving_altering_mapping():
    pred_filename = './data/submission/results_cot_greedy.json'
    gold_filename = './data/gold_test.json'

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)

    Overall_F1, Overall_Rec, Overall_Prec = F1_Recall_Precision(predictions, gold)
    print(f'Overall f1: {Overall_F1}')
    print(f'Overall Rec: {Overall_Rec}')
    print(f'Overall Prec: {Overall_Prec}')
    # Control Test Set F1, Recall, Precision PUBLIC
    control_set = extract_control_set(predictions, gold)

    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    print(f'predictions preserving: {len(predictions_preserving)}')
    print(f'predictions altering: {len(predictions_altering)}')
    base_mapping_preserving = {}
    base_mapping_altering = {}
    for key in list(control_set.keys()):
        base_mapping_preserving[key] = []
        base_mapping_altering[key] = []
    for key in list(predictions_preserving.keys()):
        if gold[key]['Causal_type'][1] in list(base_mapping_preserving.keys()):
            base_mapping_preserving[gold[key]['Causal_type'][1]].append(key)
    for key in list(predictions_altering.keys()):
        if gold[key]['Causal_type'][1] in list(base_mapping_altering.keys()):
            base_mapping_altering[gold[key]['Causal_type'][1]].append(key)
    preserving_length = [len(list(set(base_mapping_preserving[key]))) for key in list(base_mapping_preserving.keys())]
    altering_length = [len(list(set(base_mapping_altering[key]))) for key in list(base_mapping_altering.keys())]
    assert sum(preserving_length) == len(predictions_preserving)
    assert sum(altering_length) == len(predictions_altering)
    with open('./data/base_mapping_preserving.json', 'w') as file:
        json.dump(base_mapping_preserving, file)
    with open('./data/base_mapping_altering.json', 'w') as file:
        json.dump(base_mapping_altering, file)


def check_preserving_errors():
    extra_text = 0
    with open('./data/base_mapping_preserving.json') as file:
        base_mapping = json.load(file)
    with open('./data/gold_test.json') as file:
        gold = json.load(file)
    with open('./data/submission/results_selfcon_cot.json') as file:
        selfcon_cot = json.load(file)
    generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    generation_keys = [list(element.keys())[0] for element in generation]
    for key in list(base_mapping.keys()):
        base_prompt = generation[generation_keys.index(key)][key]['prompt']
        base_statement = generation[generation_keys.index(key)][key]['info']['Statement']
        base_output = generation[generation_keys.index(key)][key]['output']
        base_output = [output[len(base_prompt) - 2:] for output in base_output]
        # print(f'base output: {base_output}')
        base_predicted_label = selfcon_cot[key]['Prediction']
        base_gold_label = gold[key]['Label']
        preserving_keys = base_mapping[key]
        prompts = []
        print(base_prompt)
        print(base_statement)
        print('**************************')
        for p_key in preserving_keys:
            prompt = generation[generation_keys.index(p_key)][p_key]['prompt']
            p_statement = generation[generation_keys.index(p_key)][p_key]['info']['Statement']
            predicted_p_label = selfcon_cot[p_key]['Prediction']
            p_output = generation[generation_keys.index(p_key)][p_key]['output']
            p_output = [output[len(prompt) - 2:] for output in p_output]
            if base_gold_label in ['Entailment', 'Contradiction'] and base_statement not in p_statement:
                extra_text += 1
                print(key)
                print(p_key)
                print(f'{base_statement}\n{p_statement}')
                '''for output in p_output:
                    print(f'output: {output}')
                    print('\n')'''
                print('===========================')
        print('\n\n')
    print(extra_text)


def get_base_key(altering, key_changed):
    base_key = None
    for key in altering.keys():
        if key_changed in altering[key]:
            base_key = key
            return base_key

    return base_key


def compare_faithfulness_selfcon_cot_cot_greedy():
    with open('./data/gold_test.json') as file:
        gold = json.load(file)
    # results_cot_greedy.json results_selfcon_cot.json results_label_only.json
    with open('./data/submission/results_cot_greedy.json') as file:
        cot = json.load(file)
    with open('./data/submission/results_selfcon_cot.json') as file:
        selfcon = json.load(file)
    with open('./data/base_mapping_altering.json') as file:
        altering = json.load(file)
    selfcon_generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    cot_generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_cot_greedy_test_fsdp_5epoch_w_dev.jsonl'))

    selfcon_generation_keys = [list(element.keys())[0] for element in selfcon_generation]
    cot_generation_keys = [list(element.keys())[0] for element in cot_generation]

    contrast_predictions = extract_contrast_set(selfcon, gold)  # 5000 interventions
    # predictions preserving 4136, predictions altering 864
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)

    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(selfcon, gold)
    # para preserving 1500
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    # cont preserving 750, cont altering 750
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    # print(list(cont_preserving.keys())[:5])
    # numerical para preserving 224
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    # print(list(numerical_para_preserving.keys())[:5])
    # numerical cont preserving 162, numerical cont altering 114
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    print('only cot wrong')
    cot_wrong = 0
    selfcon_wrong = 0
    both_wrong = 0

    for key in cont_altering.keys():
        if selfcon[key]['Prediction'] == gold[key]['Label'] and cot[key]['Prediction'] != gold[key]['Label']:
            # print(key)
            cot_wrong += 1
            print(f'gold label: {gold[key]["Label"]}')
            for _, value in cot_generation[cot_generation_keys.index(key)].items():
                prompt = value['prompt']
                print(prompt)
                print(value['output'][len(prompt) - 2:])
            print('Self Consistent')
            for _, value in selfcon_generation[selfcon_generation_keys.index(key)].items():
                prompt = value['prompt']
                for o in value['output']:
                    print(o[len(prompt) - 2:])
                    print('=====================')

            print('#######################')
    print('both wrong')
    for key in cont_altering.keys():
        if selfcon[key]['Prediction'] != gold[key]['Label'] and cot[key]['Prediction'] != gold[key]['Label']:
            # print(key)
            both_wrong += 1
            '''print(f'gold label: {gold[key]["Label"]}')
            for _, value in cot_generation[cot_generation_keys.index(key)].items():
                prompt = value['prompt']
                print(prompt)
                print(value['output'][len(prompt) - 2:])
            print('Self Consistent')
            for _, value in selfcon_generation[selfcon_generation_keys.index(key)].items():
                prompt = value['prompt']
                for o in value['output']:
                    print(o[len(prompt) - 2:])
                    print('=====================')'''
    print('########################')
    print('selfcon wrong')
    for key in cont_altering.keys():
        if selfcon[key]['Prediction'] != gold[key]['Label'] and cot[key]['Prediction'] == gold[key]['Label']:
            # print(key)
            selfcon_wrong += 1
    print(f'cot wrong {cot_wrong}')
    print(f'selfcon wrong {selfcon_wrong}')
    print(f'both wrong {both_wrong}')


def check_faithfulness_errors():
    with open('./data/gold_test.json') as file:
        gold = json.load(file)
    # results_cot_greedy.json results_selfcon_cot.json results_label_only.json
    with open('./data/submission/results_cot_greedy.json') as file:
        selfcon_cot_labels = json.load(file)
    with open('./data/base_mapping_altering.json') as file:
        altering = json.load(file)
    selfcon_cot_generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))

    selfcon_cot_keys = [list(element.keys())[0] for element in selfcon_cot_generation]
    contrast_predictions = extract_contrast_set(selfcon_cot_labels, gold)  # 5000 interventions
    # predictions preserving 4136, predictions altering 864
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)

    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(selfcon_cot_labels, gold)
    # para preserving 1500
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    # cont preserving 750, cont altering 750
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    # print(list(cont_preserving.keys())[:5])
    # numerical para preserving 224
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    # print(list(numerical_para_preserving.keys())[:5])
    # numerical cont preserving 162, numerical cont altering 114
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)
    inconsistent = 0
    base_keys = []
    num_cont = 0
    cont_correct = 0
    for key in cont_altering.keys():
        if selfcon_cot_labels[key]['Prediction'] == gold[key]['Label']:
            cont_correct += 1
        num_cont += 1
        base_key = get_base_key(altering, key)
        if base_key not in base_keys:
            base_keys.append(base_key)
        # print(f'{key}#{base_key}')
        '''if selfcon_cot_labels[base_key]['Prediction'] == gold[base_key]['Label'] and selfcon_cot_labels[key]['Prediction'] != \
                gold[key]['Label']:
            # print(f'base label: {gold[base_key]["Label"]}')
            # print(f'altered label: {gold[key]["Label"]}')
            for _, value in selfcon_cot_generation[selfcon_cot_keys.index(base_key)].items():
                prompt = value['prompt']
                print(f'prompt: {prompt}')
                for o in value['output']:
                    print(o[len(prompt) - 2:])
                    print('===========================')
                print(f'statement: {value["info"]["Statement"]}')
            for _, value in selfcon_cot_generation[selfcon_cot_keys.index(key)].items():
                prompt = value['prompt']
                print(f'statement: {value["info"]["Statement"]}')
                for o in value['output']:
                    print(o[len(prompt) - 2:])
                    print('++++++++++++++++++++++++++++')
            print('##########################')'''
    print(f'numerical altering: {num_cont}')
    print(f'base keys: {len(base_keys)}')
    print(f'inconsistent {inconsistent}')
    base_correct = 0
    for key in base_keys:
        if selfcon_cot_labels[key]['Prediction'] == gold[key]['Label']:
            base_correct += 1
    print(f'base accuracy {base_correct / len(base_keys)}')
    print(f'cont accuracy: {cont_correct / num_cont}')
    cont_Faithfulness = faithfulness(cont_altering, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    Faithfulness = faithfulness(predictions_altering, gold)
    print(f'faithfulness: {Faithfulness}')
    print(f'contradiction faithfulness: {cont_Faithfulness}')
    print(f'numerical count faithfulness: {numerical_cont_Faithfulness}')


def check_baseline_errors():
    with open('./data/gold_test.json') as file:
        gold = json.load(file)
    with open('./data/submission/results_selfcon_cot.json') as file:
        selfcon_cot_labels = json.load(file)
    selfcon_cot_generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    selfcon_cot_keys = [list(element.keys())[0] for element in selfcon_cot_generation]
    control_predictions = extract_control_set(selfcon_cot_labels, gold)
    num_errors = 0
    idx = 0
    sections = {"Results": [], "Eligibility": [], "Intervention": [], "Adverse Events": []}
    sections_wrong = {"Results": 0, "Eligibility": 0, "Intervention": 0, "Adverse Events": 0}

    for key in list(control_predictions.keys()):
        section = gold[key]['Section_id']
        prediction = selfcon_cot_labels[key]['Prediction']
        if prediction not in ['Entailment', 'Contradiction']:
            print(prediction)
        gold_label = gold[key]['Label']
        sections[section].append((gold_label, prediction))

        if gold[key]['Label'] != selfcon_cot_labels[key]['Prediction']:
            section = gold[key]['Section_id']
            sections_wrong[gold[key]['Section_id']] += 1
            print(idx)
            if section == 'Results':
                print(f'gold label: {gold[key]["Label"]}')
                print(f'predicted label: {selfcon_cot_labels[key]["Prediction"]}')
                generation = selfcon_cot_generation[selfcon_cot_keys.index(key)][key]
                prompt = generation['prompt']
                output = [element[len(prompt) - 2:] for element in generation['output']]
                print(prompt)
                for p in output:
                    print(p)
                    print('\n')
            print('############################')

            idx += 1
    total_predictions = []
    total_gold_labels = []
    for key in sections.keys():
        print(key)
        prediction_gold = sections[key]
        gold_labels = [1 if element[0] == 'Entailment' else 0 for element in prediction_gold]
        predictions = [1 if element[1] == 'Entailment' else 0 for element in prediction_gold]
        assert len(gold_labels) == len(predictions)
        total_predictions.extend(predictions)
        total_gold_labels.extend(gold_labels)
        F1 = f1_score(gold_labels, predictions)
        print(F1)
    print('Total')
    F1 = f1_score(total_gold_labels, total_predictions)
    print(F1)
    # print(sections_wrong)
    # print(sections)


def numerical_para_preserving_consistency():
    pred_filename = './data/submission/results_selfcon_cot.json'
    gold_filename = './data/gold_test.json'
    generation_filename = \
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)
    generation = list(jsonlines.open(generation_filename))
    generation_keys = []
    generation_values = []
    for element in generation:
        for key, value in element.items():
            generation_keys.append(key)
            generation_values.append(value)
    generation = dict(zip(generation_keys, generation_values))

    # Intervention-wise Consistency & Faithfullness HIDDEN
    # para predictions 1500, cont predictions 1500, numerical para predictions 224, numerical cont predictions 276,
    # definitions predictions 1500
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    # para preserving 1500
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    # cont preserving 750, cont altering 750
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    # numerical para preserving 224
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    both_correct = 0
    base_correct = 0
    both_wrong = 0
    para_preserving_correct = 0
    base_keys = []
    for key in list(numerical_para_preserving.keys()):
        base_key = gold[key]['Causal_type'][1]
        if base_key not in base_keys:
            base_keys.append(base_key)

        if gold[base_key]['Label'] == predictions[base_key]['Prediction'] and gold[key]['Label'] == predictions[key][
            'Prediction']:
            both_correct += 1
        elif gold[base_key]['Label'] == predictions[base_key]['Prediction'] and gold[key]['Label'] != predictions[key][
            'Prediction']:
            print(f'Count {base_correct}')
            base_correct += 1
            prompt = generation[base_key]['prompt']
            base_statement = generation[base_key]['info']['Statement']
            base_output = generation[base_key]['output']
            base_output = [output[len(prompt) - 2:] for output in base_output]
            para_statement = generation[key]['info']['Statement']
            para_output = generation[key]['output']
            para_output = [output[len(prompt) - 2:] for output in para_output]
            print(prompt)
            print('--------------------------------------------')
            print(f'gold label: {gold[base_key]["Label"]}')
            print(f'base statement: {base_statement}')
            for output in base_output:
                print(output)
                print('\n')
            print('--------------------------------------------')
            print(f'para statement: {para_statement}')
            for output in para_output:
                print(output)
                print('\n')

        elif gold[base_key]['Label'] != predictions[base_key]['Prediction'] and gold[key]['Label'] == predictions[key][
            'Prediction']:
            para_preserving_correct += 1

        else:
            both_wrong += 1
    print(f'both correct {both_correct}')
    print(f'base correct {base_correct}')
    print(f'para preserving correct {para_preserving_correct}')
    print(f'both wrong {both_wrong}')
    print(f'total {both_correct + base_correct + para_preserving_correct + both_wrong}')
    base_correct = 0
    for key in base_keys:
        if predictions[key]['Prediction'] == gold[key]['Label']:
            base_correct += 1

    print(f'base keys {len(base_keys)}-{base_correct} correct')


def main():
    # Load files
    '''input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    pred_dir = os.path.join(input_dir, 'res')
    gold_dir = os.path.join(input_dir, 'ref')

    if not os.path.isdir(pred_dir):
        raise RuntimeError('{} does not exist'.format(pred_dir))

    if not os.path.isdir(gold_dir):
        raise RuntimeError('{} does not exist'.format(gold_dir))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    gold_filename = os.path.join(gold_dir, 'gold_test.json')
    pred_filename = os.path.join(pred_dir, 'results.json')'''
    pred_filename = './data/submission/results_label_only.json'
    gold_filename = './data/gold_test.json'
    print(pred_filename)
    generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    generation_keys = [list(element.keys())[0] for element in generation]

    with open(pred_filename) as json_file:
        predictions = json.load(json_file)

    with open(gold_filename) as json_file:
        gold = json.load(json_file)

    # Overall information
    Overall_F1, Overall_Rec, Overall_Prec = F1_Recall_Precision(predictions, gold)
    print(f'Overall f1: {Overall_F1}')
    print(f'Overall Rec: {Overall_Rec}')
    print(f'Overall Prec: {Overall_Prec}')

    # Control Test Set F1, Recall, Precision PUBLIC
    Control_F1, Control_Rec, Control_Prec = F1_Recall_Precision(extract_control_set(predictions, gold), gold)

    # Contrast Consistency & Faithfullness PUBLIC
    contrast_predictions = extract_contrast_set(predictions, gold)  # 5000 interventions
    # predictions preserving 4136, predictions altering 864
    predictions_preserving, predictions_altering = extract_by_causal_type(contrast_predictions, gold)
    # print(f'predictions_preserving: {len(predictions_preserving)}')
    # print(f'predictions altering: {len(predictions_altering)}')

    Faithfulness = faithfulness(predictions_altering, gold)
    Consistency = consistency(predictions_preserving, gold)

    # Intervention-wise Consistency & Faithfullness HIDDEN
    # para predictions 1500, cont predictions 1500, numerical para predictions 224, numerical cont predictions 276,
    # definitions predictions 1500
    para_predictions, cont_predictions, numerical_para_predictions, numerical_cont_predictions, definitions_predictions = \
        extract_by_intervention(predictions, gold)
    # para preserving 1500
    para_preserving = extract_by_causal_type(para_predictions, gold)[0]
    # cont preserving 750, cont altering 750
    cont_preserving, cont_altering = extract_by_causal_type(cont_predictions, gold)
    # print(list(cont_preserving.keys())[:5])
    # numerical para preserving 224
    numerical_para_preserving = extract_by_causal_type(numerical_para_predictions, gold)[0]
    # print(list(numerical_para_preserving.keys())[:5])
    # numerical cont preserving 162, numerical cont altering 114
    numerical_cont_preserving, numerical_cont_altering = extract_by_causal_type(numerical_cont_predictions, gold)

    # definitions preserving 1500
    definitions_preserving = extract_by_causal_type(definitions_predictions, gold)[0]
    '''tmp_keys = list(definitions_preserving.keys())[:10]
    for key in tmp_keys:
        for _, value in generation[generation_keys.index(key)].items():
            print(value['prompt'])
            print('############')'''

    para_Consistency = consistency(para_preserving, gold)
    cont_Faithfulness = faithfulness(cont_altering, gold)
    cont_Consistency = consistency(cont_preserving, gold)
    numerical_para_Consistency = consistency(numerical_para_preserving, gold)
    numerical_cont_Faithfulness = faithfulness(numerical_cont_altering, gold)
    numerical_cont_Consistency = consistency(numerical_cont_preserving, gold)
    definitions_Consistency = consistency(definitions_preserving, gold)

    # Intervention-wise F1, Recall, Precision HIDDEN
    Contrast_F1, Contrast_Rec, Contrast_Prec = F1_Recall_Precision(contrast_predictions, gold)
    para_F1, para_Rec, para_Prec = F1_Recall_Precision(para_predictions, gold)
    cont_F1, cont_Rec, cont_Prec = F1_Recall_Precision(cont_predictions, gold)

    numerical_para_F1, numerical_para_Rec, numerical_para_Prec = F1_Recall_Precision(numerical_para_predictions, gold)
    numerical_cont_F1, numerical_cont_Rec, numerical_cont_Prec = F1_Recall_Precision(numerical_cont_predictions, gold)
    definitions_F1, definitions_Rec, definitions_Prec = F1_Recall_Precision(definitions_predictions, gold)

    # Output results

    print(f'Control_F1: {Control_F1}')
    print(f'Control_Recall: {Control_Rec}')
    print(f'Control_Precision: {Control_Prec}')
    print(f'Contrast_F1: {Contrast_F1}')
    print(f'Contrast_Recall: {Contrast_Rec}')
    print(f'Contrast_Precision: {Contrast_Prec}')
    print(f'Faithfulness: {Faithfulness}')
    print(f'Consistency: {Consistency}')
    print(f'Para_Consistency: {para_Consistency}')
    print(f'Cont_Faithfulness: {cont_Faithfulness}')
    print(f'Cont_Consistency: {cont_Consistency}')
    print(f'Numerical_Para_Consistency: {numerical_para_Consistency}')
    print(f'Numerical_Cont_Faithfulness: {numerical_cont_Faithfulness}')
    print(f'Numerical_Cont_Consistency: {numerical_cont_Consistency}')
    print(f'Definitions_Consistency: {definitions_Consistency}')
    print(f'Para_F1: {para_F1}')
    print(f'Para_Recall: {para_Rec}')
    print(f'Para_Precision: {para_Prec}')
    print(f'Cont_F1: {cont_F1}')  # 0
    print(f'Cont_Recall: {cont_Rec}')  # 0
    print(f'Cont_Precision: {cont_Prec}')  # 0
    print(f'Numerical_Para_F1: {numerical_para_F1}')
    print(f'Numerical_Para_Recall: {numerical_para_Rec}')
    print(f'Numerical_Para_Precision: {numerical_para_Prec}')
    print(f'Numerical_Cont_F1: {numerical_cont_F1}')  # 0
    print(f'Numerical_Cont_Recall:  {numerical_cont_Rec}')  # 0
    print(f'Numerical_Cont_Precision: {numerical_cont_Prec}')  # 0
    print(f'Definitions_F1: {definitions_F1}')
    print(f'Definitions_Recall: {definitions_Rec}')
    print(f'Definitions_Precision: {definitions_Prec}')


def search_for_preserving_altering_examples():
    with open('./data/gold_test.json') as file:
        gold = json.load(file)
    generation = list(jsonlines.open(
        './data/prompts/7B/mixtral_lora_7B_selfcon_cot_test_fsdp_5epoch_w_dev.jsonl'))
    generation_keys = [list(element.keys())[0] for element in generation]
    with open('./data/base_mapping_preserving.json') as file:
        preserving = json.load(file)
    with open('./data/base_mapping_altering.json') as file:
        altering = json.load(file)
    keys_intersection = []
    for key in preserving.keys():
        if len(preserving[key]) > 0 and len(altering[key]) > 0:
            keys_intersection.append(key)
    count = 0
    for key in keys_intersection:
        preserving_example = preserving[key]
        intervention_types = list(set([gold[tmp]['Intervention'] for tmp in preserving_example]))
        altering_example = altering[key]
        if len(intervention_types) > 1 and 'Paraphrase' in intervention_types:
            for _, value in generation[generation_keys.index(key)].items():
                print(value['prompt'])
            for example_key in preserving_example:
                print(example_key)
                print(gold[example_key]['Intervention'])
                for _, value in generation[generation_keys.index(example_key)].items():
                    print(value['info']['Statement'])
            print('Altering')
            for example_key in altering_example:
                print(example_key)
                for _, value in generation[generation_keys.index(example_key)].items():
                    print(value['info']['Statement'])
            print('###############')
    print(count)


if '__main__' == __name__:
    compare_faithfulness_selfcon_cot_cot_greedy()
    # check_faithfulness_errors()
    # search_for_preserving_altering_examples()
    # get_preserving_altering_mapping()
    # check_baseline_errors()
    # main()
    # numerical_para_preserving_consistency()
