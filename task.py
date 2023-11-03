from transformers import AutoModelForSequenceClassification, AutoModelForTokenClassification ,AutoModelForMaskedLM

def model_dataset(args) :
    if args.task == 'nli' :
        # Entailment : 0 / Neutral : 0 / Contradiction : 0
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 3)
        args.dataset = 'xnli'
        if args.fdataset != None:
            args.dataset = args.fdataset
    elif args.task == 'ner' :
        # O : 0, B-PER : 1, I-PER : 2, B-ORG : 3, I-ORG : 4, B-LOC : 5, I-LOC : 6.
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels = 7)
        if args.fdataset == None:
            args.dataset = 'wikiann'
        else:
            args.dataset = args.fdataset
    elif args.task == 'cls':
        # True / False
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 2)
        args.dataset = 'paws-x'
    elif args.task == 'pos':
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels = 17)
        if args.fdataset == None:
            args.dataset = 'wietsedv/udpos28'
        else:
            args.dataset = args.fdataset
    else:
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        if args.fdataset == None:
            args.dataset = 'wikiann'
        else:
            args.dataset = args.fdataset
        
    return model, args
def mask_model_dataset(args) :
    if args.mask_task == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
        args.dataset = args.fdataset
    elif args.mask_task == 'nli' :
        # Entailment : 0 / Neutral : 0 / Contradiction : 0
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 3)
        args.dataset = 'xnli'
    elif args.mask_task == 'ner' :
        # O : 0, B-PER : 1, I-PER : 2, B-ORG : 3, I-ORG : 4, B-LOC : 5, I-LOC : 6.
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels = 7)
        if args.fdataset == None:
            args.dataset = 'wikiann'
        else:
            args.dataset = args.fdataset
    elif args.mask_task == 'paws-x':
        # True / False
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels = 2)
        args.dataset = 'paws-x'
    elif args.mask_task == 'pos':
        # 17 labels
        model = AutoModelForTokenClassification.from_pretrained(args.model_name, num_labels = 17)
        if args.fdataset == None:
            args.dataset = 'wietsedv/udpos28'
        else:
            args.dataset = args.fdataset    

    return model, args