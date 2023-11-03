import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shot',type=float,required=True)
    parser.add_argument('--p',type=float,required=False)
    parser.add_argument('--task',type=str,required=True)
    parser.add_argument('--range',type=int,required=False,default=12)
    parser.add_argument('--mask',type=str,required=False,default='other')
    parser.add_argument('--device',type=str,required=False,default='0')
    parser.add_argument('--seed',type=int,required=False)
    parser.add_argument('--model_name',type=str,default='xlm-roberta-base')
    parser.add_argument('--dataset',type=str,default='wikiann')
    args = parser.parse_args()
    source_mask = {}
    target_mask = {}
    
    if args.task =='ner':
        source_language = ['en','ar','el','es','fi','fr','he','id','it','ja','ko','ru','sv','th','tr','vi','zh']
        target_language = ['en','ar','el','es','fi','fr','he','id','it','ja','ko','ru','sv','th','tr','vi','zh']
    elif args.task == 'cls':
        source_language = ['en', 'de', 'es', 'fr' ,'ja' ,'ko' ,'zh']
        target_language = ['en', 'de', 'es', 'fr' ,'ja' ,'ko' ,'zh']
    elif args.task =='nli':
        source_language = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
        target_language = ['en', 'ar', 'bg', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    else:
        source_language = ['en','bg','es','fi','fr','hi','id','it','ja','ko','nl','no','pl','pt','ru','sk','sv','tr','uk','zh']
        target_language = ['en','bg','es','fi','fr','hi','id','it','ja','ko','nl','no','pl','pt','ru','sk','sv','tr','uk','zh']
    
    df = pd.DataFrame(index=source_language,columns=target_language)
    device = 'cuda:'+args.device
    
    mask={}

    for source in source_language:
        with open(f'./output/mask/[mlm_{args.mask}]{source}_{args.dataset}_{args.model_name}_{int(args.shot)}_{args.seed}.pickle','rb') as f:
            mask[source] = pickle.load(f)
        # TODO: pytorch: torch.kthvalue
        r = None
        for k, v in mask[source].items():
            v = v.view(-1).detach().cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)

        polar = np.percentile(r, (1-args.p)*100)
        for k in mask[source]:
            mask[source][k] = mask[source][k] >= polar
        print('Polar => {}'.format(polar))
        
    for source in tqdm(source_language):
        for target in tqdm(target_language):

            count=0
            total=0
            for k in mask[source]:
                a = mask[source][k].to(device)
                b = mask[target][k].to(device)
                
                c = (a|b).cpu().numpy()   
                temp = np.unique(c,return_counts=True)
                for i in range(len(temp[0])):
                    if temp[0][i]:
                        total+=temp[1][i]
                d = (a&b).cpu().numpy() 
                temp2 = np.unique(d,return_counts=True)
                for i in range(len(temp2[0])):
                    if temp2[0][i]:
                        count+=temp2[1][i]
            df.loc[source,target] = count/total

    df.to_csv(f'./output/output_csv/{args.model_name}/{args.mask}/{args.task}_{args.mask}_{args.dataset}_{int(args.shot)}_{args.p}_{args.seed}_overlap.csv')

if __name__ == "__main__":
    main()