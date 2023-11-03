import random
import evaluate
import collections
import argparse

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def preprocess_validation_examples(examples):
    max_length = 384
    stride = 128

    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    '''
    After tokenizing, there are two things to do
        1. add 'example_id' column to dataset
            inputs['example_id'][i] is a string id of i-th feature's example.
            it refers the example this feature originates from. 
        2. change 'offset_mapping' 
            original inputs['offset_mapping'][i] contains span(start_idx, end_idx) in raw context for each tokens in i-th feature.
            after changing, span of question tokens becomes (-1, -1). but span of context remains same.
    '''
    
    example_ids = []
    example_map = inputs.pop("overflow_to_sample_mapping")
    null_offset = (-1, -1) # not a context token

    # inputs.sequence_ids(i) returns a list which tells whether the tokens of i-th feature belong to question or context
    # ex) inputs.sequence_ids(i) = [0, 0, 0, 1, 1, 1]
    #  => first three tokens belongs to question and last three tokens belongs to context
    for i in range(len(inputs["input_ids"])):
        example_idx = example_map[i]
        example_ids.append(examples["id"][example_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else null_offset for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def make_preprocessed_test_dataset(dataset, preprocess_func):
    dataset = dataset.map(preprocess_func, batched=True, batch_size=128, remove_columns=dataset.column_names)
    dataset.set_format(type='torch')
    return dataset


def compute_metrics(args, start_logits, end_logits, features, examples):
    max_answer_length = 30
    null_offset = (-1, -1) # not context token

    metric = evaluate.load("squad")

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # iterate over all features which belong to one example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = torch.argsort(start_logit, descending=True)[:args.n_best]
            end_indexes = torch.argsort(end_logit, descending=True)[:args.n_best]

            # find the best answer from a (start, end) probability array driven by one feature
            for start_index in start_indexes:
                for end_index in end_indexes:

                    # skip answers that are not fully in the context
                    if offsets[start_index] == null_offset or offsets[end_index] == null_offset:
                        continue
                    # skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)
        
        # after the second inner for-loop for example,
        # anwers = [
        #   {'text': '22 years old', 'logit_score': 1.8} => best answer!
        #   {'text': '22 years', 'logit_score': 1.5} 
        #   {'text': '22', 'logit_score': 1.0}
        # ]
        #
        # these are candidate answers from one example

        # select the best answer by the largest logit_score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

        # for example,
        # predicted_answers = [
        #   {'id': 1, 'prediction_text' : '22 years old'}
        # ]

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


def test(args, model, test_dataset, preprocessed_test_dataset, performances, lang):
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    
    start = 0
    num_features = len(preprocessed_test_dataset[lang])
    batch_size = 64

    exact_match = 0
    f1_score = 0
    
    model.eval()

    while start < num_features:
        if start + batch_size <= num_features:
            end = start + batch_size
        else:
            end = num_features

        features = preprocessed_test_dataset[lang].select(range(start, end))
        examples = test_dataset[lang]['validation'].filter(lambda example: example['id'] in features['example_id'])

        batch = {k: features[k].to(device) for k in features.column_names if k not in ['offset_mapping', 'example_id']}
        outputs = model(**batch)

        start_logits = outputs['start_logits']
        end_logits = outputs['end_logits']

        metrics = compute_metrics(args, start_logits, end_logits, features, examples) # returns average performance
        
        exact_match += metrics['exact_match'] * len(examples)
        f1_score += metrics['f1'] * len(examples)
        
        start += batch_size

    num_examples = len(test_dataset[lang]['validation']) # NOTE: len(test_dataset[lang]) = 1
    exact_match /= num_examples
    f1_score /= num_examples

    print(f'Exact Match: {exact_match}, F1 Score: {f1_score}')

    performances[lang] = {'exact_match': exact_match, 'f1_score': f1_score}


def main():
    # commandline argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, required=True, default='0')
    parser.add_argument('--model', type=str, required=False, default='xlm-roberta-base')
    parser.add_argument('--checkpoint', type=str, required=True, default='model1.pth')
    parser.add_argument('--save_to', type=str, required=True, default='result1.txt')
    parser.add_argument(
        "--n_best",
        default=20,
        type=int,
        help='number of candidates(tuple of start and end logit) considering for inference'
    )
    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # model & tokenizer
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    model.load_state_dict(torch.load('model/' + args.checkpoint))
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # XQuAD
    xquad_split = ['en', 'zh', 'tr', 'ru', 'ar', 'hi', 'vi', 'th', 'es', 'el', 'de']
    xquad_dataset = {}
    for lang in xquad_split:
        xquad_dataset[lang] = load_dataset('xquad', 'xquad.' + lang)
    
    preprocessed_xquad_dataset = {}
    for lang in xquad_split:
        preprocessed_xquad_dataset[lang] = make_preprocessed_test_dataset(
            xquad_dataset[lang]['validation'], preprocess_validation_examples
        )

    # test
    performances = {}
    for lang in xquad_split:
        print(f'target language: {lang}')
        test(args, model, xquad_dataset, preprocessed_xquad_dataset, performances, lang)
        print()
    print('Done!')
    print(performances)

    # save result
    with open(f'.output/output_csv/{args.seed}/' + args.save_to, 'w') as f:
        for lang in performances:
            string = f'"{lang}"' + ': ' + str(performances[lang]) + ',' + '\n'
            f.write(string)


if __name__ == '__main__':
    main()