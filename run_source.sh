task=ner
lan=en
seed=42
device=0

python source.py --task $task --seed $seed  --device $device --source $lan 

# all language for ner
# for lan in en ar el es fi fr he id it ja ko ru sv th tr vi zh
# do 
#     echo $lan
#     python source.py --task ner --seed 42  --device 0 --source $lan &
#     wait
# done