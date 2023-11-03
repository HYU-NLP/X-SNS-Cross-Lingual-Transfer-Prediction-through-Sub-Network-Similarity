from args import *
from mask import *
from task import *
from util import *
from train import *
from dataloader import *

import pandas as pd
import copy
import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def target() :
    
    args = parser_args()
    seed_everything(args.seed)

    args.model_checkpoint = True
    device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'
    if args.task == 'cls':
        language = ['en','fr','es','de','zh','ja','ko']
    elif args.task == 'nli':
        language = ['en','ar','bg','de','el','es','fr','hi','ru','sw','th','tr','ur','vi','zh']
    elif args.task == 'ner':
        language = ['en','ar','el','es','fi','fr','he','id','it','ja','ko','ru','sv','th','tr','vi','zh']
        target_lang = ['en','ar','el','es','fi','fr','he','id','it','ja','ko','ru','sv','th','tr','vi','zh']
        if args.low_resource:
            target_lang = ['af','br','ckb','fy','ga','hi','is','kk','lb','mr','sq','sw','te','tt','uz']
    elif args.task == 'pos':
        language = ['en','bg','es','fi','fr','hi','id','it','ja','ko','nl','no','pl','pt','ru','sk','sv','tr','uk','zh']

    if args.task == 'ner':
        target_lang = target_lang
    else:
        target_lang = language
    source_language = language

    acc =[]
    df = pd.DataFrame(index=source_language,columns=target_lang)
    zero_shot = pd.DataFrame(index=source_language,columns=target_lang)
    target_data = {}
    target_test = {}
    target_val = {}
    model, args = model_dataset(args)
    if args.task == 'pos':
        for target in target_lang:
            seed_everything(args.seed)
            target_train_dataset = load_dataset('wietsedv/udpos28', target, split = 'train')
            target_data[target] = target_train_dataset
            target_val_dataset = load_dataset('wietsedv/udpos28', target, split = 'validation')
            target_val[target] = target_val_dataset
            target_test[target] = load_dataset('wietsedv/udpos28', target, split = 'test')
    else:
         for target in target_lang:
            seed_everything(args.seed)
            target_train_dataset = load_dataset(args.dataset, target, split = 'train')
            target_data[target] = target_train_dataset
            target_val_dataset = load_dataset(args.dataset, target, split = 'validation')
            target_val[target] = target_val_dataset
            target_test[target] = load_dataset(args.dataset, target, split = 'test')
    for source in source_language:
        print(f'#######{source}#########')
        for target in target_lang:
            seed_everything(args.seed)
            print(target)
            model, args = model_dataset(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            optimizer = AdamW(model.parameters(), lr = args.learning_rate)
            model.to(device)
            torch.cuda.empty_cache()
            if args.model_checkpoint:
                print('loading model checkpoint...')
                print(f"load  [{args.task}]{args.model_name}_{source}_{args.model_ver}.pt")
                checkpoint = torch.load(f"{args.model_checkpoint_path}/{args.task}{args.seed}/[{args.task}]{args.model_name}_{source}_{args.model_ver}.pt",map_location='cpu')
                model.load_state_dict(checkpoint['model_state_dict'],)
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print('loading model checkpoint finished.')
            
            train_dataset_size = len(target_data[target])
            val_dataset_size = len(target_val[target])
            test_dataset_size = len(target_test[target])
            print(f"Target train dataset size : {train_dataset_size}")
            print(f"Target score dataset size : {test_dataset_size}")

            target_test_dataloader = make_dataloader(args.task, target_test[target], tokenizer, args.batch_size, False) 
            best_test = 0
            best_epoch=0
            zeroshot_accuracy = validation(args, model, target_test_dataloader, device, test_dataset_size)
            zero_shot.loc[source][target] = float(zeroshot_accuracy)
            print(float(zeroshot_accuracy))
   
    print(df)
    if args.low_resource:
        zero_shot.to_csv(f'./output/output_csv/{args.seed}/{args.task}_zeroshot({args.model_name})low_resource{args.device}.csv')
    else:
        zero_shot.to_csv(f'./output/output_csv/{args.seed}/{args.task}_zeroshot({args.model_name}){args.device}.csv')

if __name__ == '__main__':
    target()