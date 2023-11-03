import argparse

def parser_args() :
    parser = argparse.ArgumentParser()

    # base setting
    parser.add_argument('--seed', type=int, required=False, default=42)
    parser.add_argument('--device', type=str, required=False, default='0')
    parser.add_argument('--tqdm', action='store_true', required=False, default=False) # nohup 사용시 False 해놓으면 편함

    # dataset setting
    parser.add_argument('--task', type=str, required=False, default='nli')
    parser.add_argument('--fdataset',type=str,default=None)
    parser.add_argument('--source', type=str, required=False, default='en')
    parser.add_argument('--target', type=str, required=False, default='zh')
    parser.add_argument('--target_tuning', action='store_true', default=False)
    parser.add_argument('--wiki',action='store_true', default=False)   

    # model setting
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--model_name', type=str, required=False, default='xlm-roberta-base')
    parser.add_argument('--epochs', type=int, default=20)

    # additional config
    parser.add_argument('--p', type=float, required=False, default=0.04)
    parser.add_argument('--embedding', action='store_true', default=False) # embedding layer freeze
    parser.add_argument('--mask', type=str, default=None) # fisher, h0, 
    parser.add_argument('--shot', type=float, default=0)
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--low_resource',action='store_true', default=False)
    parser.add_argument('--mask_task',type=str,default='mlm')
    parser.add_argument('--layer',type=int,default=12)
    # model checkpoint
    parser.add_argument('--model_checkpoint', action='store_true', default=False) # Source Training - False / Target Training - True
    parser.add_argument('--model_checkpoint_path', type=str, default='.output/model')
    parser.add_argument('--model_ver', type=str, default='full')
    
    # infer
    parser.add_argument('--fewshot',type=int,default=1)
    
    args = parser.parse_args()

    return args