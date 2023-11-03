from args import *
from task import *
from util import *
from dataloader import *

import gc ,itertools, os
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForMaskedLM ,AutoModelForTokenClassification


# Calculate Fisher Information for different parameters
def calculate_fisher(args,model,train_loader,device):
        '''
        Calculate Fisher Information for different parameters
        '''
        gradient_mask = dict()
        model = model
        model.train()

        for name, params in model.named_parameters():
            if 'layer' in name and 'lm_head.layer_norm.'not in name:
                gradient_mask[name] = params.new_zeros(params.size())
       
        # Now begin
        train_dataloader = train_loader
        
        N = len(train_dataloader)

        for inputs in tqdm(train_dataloader):
            input = {k:v.to(device) for k, v in inputs.items()}
            outputs = model(**input)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss.backward()

            for name, params in model.named_parameters():
                if 'layer' in name and 'lm_head.layer_norm.'not in name:
                    torch.nn.utils.clip_grad_norm_(params, 1)
                    gradient_mask[name] += (params.grad ** 2) / N
            model.zero_grad()
        
        return gradient_mask


def make_mask() :
    args = parser_args()
    seed_everything(args.seed)
    device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.mask_task == 'nli' : 
        args.language_list = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    elif args.mask_task  == 'cls' :
        args.language_list = ['en','de', 'es', 'fr', 'ja', 'ko', 'zh']
    elif args.mask_task  == 'ner' :
        args.language_list = ['en','ar','el','es','fi','fr','he','id','it','ja','ko','ru','sv','th','tr','vi','zh']
    elif args.mask_task == 'pos' :
        args.language_list = ['en','bg','es','fi','fr','hi','id','it','ja','ko','nl','no','pl','pt','ru','sk','sv','tr','uk','zh']
    elif args.mask_task == 'qa':
        args.language_list =[]
    else :
        args.language_list = [args.source]
    mask={}
    model, args = mask_model_dataset(args)

    model.to(device)

    np.random.seed(args.seed) 

    for lan in args.language_list :
        seed_everything(args.seed)
        data = load_dataset(args.dataset, lan, split='train')
        
        data_size = len(data)
        print(f"data_size : {data_size}")

        if args.shot == 0 or args.shot == 'all':
            args.shot == 'all'
            shot_data = data
        else:
            shot_data = mask_shot_dataset(args, data)
            
        data_size = len(shot_data)
        print(f"{lan} data_size : {data_size}")

        data_loader = mask_loader(args, shot_data, tokenizer, args.batch_size, True)

        mask[lan]=calculate_fisher(args,model, data_loader, device)

        if args.dataset == 'wietsedv/udpos28':
            with open(f"output/mask/[{args.mask_task}_{args.mask}]{lan}_ud_{args.model_name}_{int(args.shot)}_{args.seed}.pickle", 'wb') as f :
                pickle.dump(mask[lan], f)
        else:
            with open(f"output/mask/[{args.mask_task}_{args.mask}]{lan}_{args.dataset}_{args.model_name}_{int(args.shot)}_{args.seed}.pickle", 'wb') as f :
                pickle.dump(mask[lan], f)
            
        gc.collect()
        torch.cuda.empty_cache()
        

    
if __name__ == '__main__' :
    make_mask()