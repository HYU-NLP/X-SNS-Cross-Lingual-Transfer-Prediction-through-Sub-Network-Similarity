import numpy as np 
from datasets import Dataset
from collections import Counter
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

def make_dataloader(task, dataset, tokenizer, batch_size, shuffle):
    # Task : NLI
    if task == 'nli' :
        def encode(examples) :
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, padding=True)
    # Task : NER
    elif task == 'ner' :
        def encode(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)

            labels = []
            for i, label in enumerate(examples[f"ner_tags"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs
    elif task == 'pos':
        def encode(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True,padding=True)

            labels = []
            for i, label in enumerate(examples[f"labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs


    if task == 'cls':
        def encode(example):
            return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding=True)
        
    
    if shuffle :
        dataset = dataset.shuffle()
    dataset = dataset.map(encode, batched=True, batch_size=1024)
    if task != 'ner' and task != 'pos' :
        dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    if task =='qa':
        dataset.set_format(type='torch')
    else:
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
    return dataloader

def shot_dataset(args, dataset) :
    if len(dataset) < args.shot:
        return dataset
    shot_data = []
    avoid_overlapping = []
    while True:
        if len(shot_data)>= args.shot:
            break
        number = np.random.randint(dataset.num_rows)
        if number in avoid_overlapping :
            continue
        avoid_overlapping.append(number)
        shot_data.append(dataset[number])
    return Dataset.from_list(shot_data)
    
def mask_shot_dataset(args, dataset) :
    if len(dataset) < args.shot:
        return dataset
    if args.mask_task == 'nli' :
        shot_dataset = []
        tmp = []
        Entailment = 0
        Neutral = 0
        Contradiction = 0
        while True :
            if (Entailment == args.shot) and (Neutral == args.shot) and (Contradiction == args.shot):
                break

            number = np.random.randint(dataset.num_rows)

            if number in tmp :
                continue
            else :
                tmp.append(number)
            
            if dataset[number]['label'] == 0 and Entailment < args.shot : # yes
                shot_dataset.append(dataset[number])
                Entailment += 1

            elif dataset[number]['label'] == 1 and Neutral < args.shot : # maybe
                shot_dataset.append(dataset[number])
                Neutral += 1

            elif dataset[number]['label'] == 2 and Contradiction < args.shot : # no
                shot_dataset.append(dataset[number])
                Contradiction += 1
        return Dataset.from_list(shot_dataset)
    

    elif args.mask_task  == 'ner' or args.mask_task  =='pos' or args.mask_task  == 'qa' or args.mask_task =='mlm':
        shot_data = []
        avoid_overlapping = []
        while True:
            if len(shot_data)>= args.shot:
                break
            number = np.random.randint(dataset.num_rows)
            if number in avoid_overlapping :
                continue
            avoid_overlapping.append(number)
            shot_data.append(dataset[number])
        return Dataset.from_list(shot_data)

    else :
        return 0
    
def mask_loader(args, dataset, tokenizer, batch_size, shuffle, shot=None) :
    if args.mask_task == 'nli' :
        def encode(examples) :
            return tokenizer(examples['premise'], truncation=True, padding=True)
    # Task CLASSIFICATION
    elif args.mask_task == 'cls' :
        def encode(examples) :
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)
    # Task : NER / POS
    elif args.mask_task == 'pos':
        def encode(examples):
            return tokenizer(examples['tokens'], truncation=True,is_split_into_words=True,padding=True)
    elif args.mask_task == 'ner' :
        def encode(examples) : 
            return tokenizer(examples['tokens'], truncation=True,is_split_into_words=True,padding=True)
    elif args.mask_task == 'mlm':
        def encode(examples) :
            if args.dataset == 'wikiann': 
                return tokenizer(examples['tokens'], truncation=True,is_split_into_words=True,padding=True)
            elif args.dataset == 'wietsedv/udpos28':
                return tokenizer(examples['tokens'], truncation=True,is_split_into_words=True,padding=True)
            elif args.dataset == 'cc100':
                return tokenizer(examples['text'], truncation=True,padding=True)
            elif args.dataset =='xnli':
                return tokenizer(examples['premise'], truncation=True, padding=True)
            elif args.dataset == 'paws-x':
                return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding=True)
    if shuffle :
        dataset = dataset.shuffle()
    #print(dataset['text'])
    dataset = dataset.map(encode, batched=True, batch_size = 1024)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    collate_fn = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=collate_fn)
    return dataloader