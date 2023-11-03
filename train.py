import torch
from tqdm import tqdm
from util import *
from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(pred,labels):

    #preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='micro',zero_division=0)
    #acc = accuracy_score(labels, pred)
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train(args, model, train_loader, optimizer, device, train_len, gradmask=None) :
    model.train()
            
    total_loss = 0
    print(f"Number of model parameters : {count_parameters(model)}")

    if args.tqdm :
        train_loader = tqdm(train_loader)

    for batch in train_loader:
        input = {k:v.to(device) for k, v in batch.items()}
        model.zero_grad()
        output = model(**input)
        loss = output['loss']
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    return total_loss / len(train_loader)


def validation(args, model, val_loader, device, val_len):
    model.eval()
    total_val = 0

    if args.tqdm :
        val_loader = tqdm(val_loader)

    with torch.no_grad():
        for batch in val_loader:
            input = {k:v.to(device) for k, v in batch.items()}
            outputs = model(**input)
            if args.task == 'ner' or args.task == 'pos':
                logits = outputs['logits']
                for logit in range(logits.shape[0]) :
                    logits_clean = logits[logit][input['labels'][logit] != -100]
                    label_clean = input['labels'][logit][input['labels'][logit] != -100].cpu()
                
                    pred = torch.argmax(logits_clean, dim=1).cpu().numpy()

                    score = compute_metrics(pred,label_clean)
                    total_val += score['f1']
            elif args.task == 'qa':
                pass
            else :
                pred = torch.argmax(outputs[1], dim=1)

                for result in pred.eq(input['labels']):
                    if result:
                        total_val += 1

    validation_score = total_val / val_len
    return validation_score

def qa_validation(args, model, test_dataset, preprocessed_test_dataset, performances, lang):
    device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
    
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

