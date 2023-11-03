from args import *
from mask import *
from task import *
from util import *
from train import *
from dataloader import *

import copy
import torch
from torch.optim import AdamW
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
import pickle

def source() :
    
    args = parser_args()
    seed_everything(args.seed)
    device = 'cuda:'+args.device if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, args = model_dataset(args)
    

    source_train_dataset = load_dataset(args.dataset, args.source, split='train')
    source_val_dataset = load_dataset(args.dataset, args.source, split='validation')


    train_dataset_size = len(source_train_dataset)
    validation_dataset_size = len(source_val_dataset)
    print(f"Train Source dataset size : {train_dataset_size}")
    print(f"Evaluate Source validation dataset size : {validation_dataset_size}")

    source_train_dataloader = make_dataloader(args.task, source_train_dataset, tokenizer, args.batch_size, True)
    source_val_dataloader = make_dataloader(args.task, source_val_dataset, tokenizer, args.batch_size, False)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    

    best_validation = 0
    best_epoch = 0
    best_loss = 0
    early_stopping = 0
    for epoch in range(args.epochs):
        # Training
        print(f"{'=' * 30} Train {epoch + 1} {'=' * 30}")
        train_loss = train(args, model, source_train_dataloader, optimizer, device, train_dataset_size)
        print(f"[Epoch {epoch+1}] Training loss : {train_loss}")
        
        # Validation
        print(f"{'=' * 30} Validation {epoch + 1} {'=' * 30}")
        validation_accuracy = validation(args, model, source_val_dataloader, device, validation_dataset_size)
        print(f"[Epoch {epoch+1}] validation f1 : {validation_accuracy}", end=' ')

        # Best Model & Early Stopping
        if best_validation < validation_accuracy :
            best_model = copy.deepcopy(model)
            best_validation = validation_accuracy
            best_epoch = epoch + 1
            best_loss = train_loss
            early_stopping = 0
            print(f"| best f1 : {best_validation} <= BEST!!")
        else :
            early_stopping += 1
            print(f"| best f1 : {best_validation}")
            if early_stopping >= 5 :
                break

    # Model Save
    
    torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': best_loss,
            'validation_f1': best_validation
        }, args.model_checkpoint_path + f"/{args.task}{args.seed}/[{args.task}]{args.model_name}_{args.source}_{args.model_ver}.pt")
        # _t({args.target}
    print(f'{args.source} Model save')


if __name__ == '__main__':
    source()