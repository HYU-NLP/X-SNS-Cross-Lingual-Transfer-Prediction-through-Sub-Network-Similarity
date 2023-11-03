# X-SNS
This is code for  "X-SNS: Cross-Lingual Transfer Prediction through Sub-Network Similarity" 

This is not the final version and will be updated soon.

## Source tuning
```python
bash run_source.sh
```
## calculate mask
```python
bash run_mask.sh
```
## calculate overlapping rate
```python
python overlap.py 
    --shot 1024 
    --p 0.15 
    --task ner
    --mask fisher 
    --device 0 
    --seed 42 
    --model_name xlm-roberta-base 
    --dataset wikiann
```
