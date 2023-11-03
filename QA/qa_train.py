import random
import time, datetime
import argparse
import collections

import numpy as np
from tqdm import tqdm
import torch
import evaluate
from datasets import load_dataset, concatenate_datasets ,Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_linear_schedule_with_warmup,
)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def preprocess_training_examples(examples):
    '''
    -examples: subset of raw dataset
    ex)
        1. question1, context1
        2. question2, context2
    '''

    max_length = 384
    stride = 128

    '''
    Tokenizer Arguments
    -truncation='only_second': if sequence length exceeds max_length(384), truncate context tokens
    -stride: size of sliding window 
    -return_offsets_mapping': returns mapping between token and the corresponding location in example
    '''
    inputs = tokenizer(
        examples['question'],
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    '''
    -inputs: preprocessed dataset returned by tokenizer
    ex)
        1. (question1)(context1_1) => list of question and context tokens from the first example
        2. (question1)(context1_2) => list of question and context tokens from the first example
        3. (question2)(context2_1) => list of question and context tokens from the second example
        4. (question2)(context2_2) => list of question and context tokens from the second example
    we call these tokenized examples as 'features'.
    as you can see, the number of original examples are 2, but the number of tokenized examples(features) are 4.
    if the number of tokens exceeds max_length, tokenizer creates several features from one example by sliding a window.
    '''

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    '''
    -offset_mapping: maps each tokens in one feature to span in example
    ex)
        i-th feature's offset mapping: offset_mapping[i] = [(0,0), (0,2), (2,5),,,]
        => first token of i-th feature doesn't exists in that example (ex, CLS, SEP token)
        => second token of i-th feature starts at 0 and ends at 1 in that example
        => third token of i-th feature starts at 2 and ends at 4 in that example
        
    -overflow_to_sample_mapping: maps features to example which they originates from
    ex)
        1. (question1)(context1_1)
        2. (question1)(context1_2)
        3. (question2)(context2_1)
        4. (question2)(context2_2)
    first two features belongs to first example.
    last two features belongs to second example.
    '''


    # start_positions[i] and end_positions[i] indicates the boundary of a consecutive context tokens (in one feature) which correspond to answer
    start_positions = []
    end_positions = []

    '''
    -start_positions and end_positions
    ex)
        question[0] = 'How old is Mary?', context[0] = 'Mary is 22 years old', answer[0] = '22 years old'
        input_ids[0]: [1(how), 2(old), 3(is), 4(Mary), 5(?), 4(Mary), 3(is), 6(22), 7(years), 2(old)]
        start_positions[0] = 7
        end_positions[0] = 9
    '''


    for i, offset in enumerate(offset_mapping):
        example_idx = sample_mapping[i] # index of example where a feature originates
        answer = answers[example_idx]
        start_char = answer["answer_start"][0] # start character index in the example
        end_char = answer["answer_start"][0] + len(answer["text"][0]) # end chararacter index in the example
        sequence_ids = inputs.sequence_ids(i) # indicates whether a token belongs to question(0) or context(1)

        # for all features, find the start and end token index which indicate a boundary of context
        # tokenizer.decode(inputs['input_ids][i][context_start: context_end]) => context!
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # if the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)


    # 'start_positions', 'end_positions': key values for labels inserted to the xlm-roberta-base model
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def make_dataloader(dataset, preprocess_func, batch_size):
    dataset = dataset.map(preprocess_func, batched=True, batch_size=128, remove_columns=dataset.column_names)
    dataset.set_format(type='torch')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train(args, model, train_dataloader):
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataloader) * args.num_epochs
    )

    train_loss = 0
    model.train()
    for idx, batch in tqdm(enumerate(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # forward pass
        outputs = model(**batch)
        loss = outputs['loss']

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss

        if (idx + 1) % 100 == 0:
            print(f'Train Loss: {train_loss / 100:>6f} [{(idx + 1) * args.batch_size} / {len(train_dataloader) * args.batch_size}]')
            train_loss = 0

        
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


def test(args, model, test_dataset, preprocessed_test_dataset,  lang):
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    
    start = 0
    num_features = len(preprocessed_test_dataset)
    batch_size = 64

    exact_match = 0
    f1_score = 0
    
    model.eval()
    with torch.no_grad():
        while start < num_features:
            if start + batch_size <= num_features:
                end = start + batch_size
            else:
                end = num_features

            features = preprocessed_test_dataset.select(range(start, end))
            examples = test_dataset[lang].filter(lambda example: example['id'] in features['example_id'])

            batch = {k: features[k].to(device) for k in features.column_names if k not in ['offset_mapping', 'example_id']}
            outputs = model(**batch)

            start_logits = outputs['start_logits']
            end_logits = outputs['end_logits']

            metrics = compute_metrics(args, start_logits, end_logits, features, examples) # returns average performance
            
            exact_match += metrics['exact_match'] * len(examples)
            f1_score += metrics['f1'] * len(examples)
            
            start += batch_size

    num_examples = len(test_dataset[lang]) # NOTE: len(test_dataset[lang]) = 1
    exact_match /= num_examples
    f1_score /= num_examples

    print(f'Exact Match: {exact_match}, F1 Score: {f1_score}')

    return exact_match

def main():
    # commandline argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_epochs',type=int,  default=5)
    parser.add_argument('--batch_size',type=int, default=16)
    parser.add_argument('--lr',type=float, default=3e-5)
    parser.add_argument('--model', type=str, default='bert-base-multilingual-cased')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument(
        "--n_best",
        default=20,
        type=int,
        help='number of candidates(tuple of start and end logit) considering for inference'
    )
    parser.add_argument('--lang', type=str, default='english')
    args = parser.parse_args()

    # seed
    seed_everything(args.seed)

    # model & tokenizer
    device = 'cuda:' + args.gpu if torch.cuda.is_available() else 'cpu'
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # dataset
    traindata = {}
    
    tempdata = load_dataset('tydiqa','secondary_task',split='train')
    for data in tempdata:
        lan = data['id'].split('-')[0]
        if lan in traindata:
            traindata[lan].append(data)
        else:
            traindata[lan] = []
    language = []
    for k in traindata:
        traindata[k] = Dataset.from_list(traindata[k])
        language.append(k)
    
    valdata ={}
    tempdata2 = load_dataset('tydiqa','secondary_task',split='validation')
    for data in tempdata2:
        #print(data)
        lan = data['id'].split('-')[0]
        #print(lan,valdata)
        if lan in valdata:
            valdata[lan].append(data)
        else:
            valdata[lan] = []
    language2 = []
    for k in valdata:
        valdata[k] = Dataset.from_list(valdata[k])
        language2.append(k)

    val = make_preprocessed_test_dataset(
            valdata[args.lang], preprocess_validation_examples
        )
    #print(len(traindata[args.lang][:2000]))
    # data preprocessing & dataloader
    train_dataloader = make_dataloader(traindata[args.lang], preprocess_training_examples, args.batch_size)

    # model
    model = AutoModelForQuestionAnswering.from_pretrained(args.model).to(device)
    print(len(traindata[args.lang]))
    print()
    for i in range(args.num_epochs):
        print(f'Epoch {i + 1}\n========================================')
        train(args, model, train_dataloader)
        val_em = test(args, model, valdata, val,args.lang)
    print('Done!')

    # save model
    torch.save(model.state_dict(), f'./output/model/qa/{args.lang}.pt')


if __name__ == '__main__':
    main()