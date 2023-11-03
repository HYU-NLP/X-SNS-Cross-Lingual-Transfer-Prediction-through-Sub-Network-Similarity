task=mlm
dataset=wietsedv/udpos28
shot=256
seed=42
device=0
mask=fisher
batch=4
lan=en

python mask.py --mask_task $task  --fdataset $dataset  --shot $seed --seed $seed --device $device --mask $mask --batch_size $batch --source $lan

# example for mlm mask at pos task
# for lan in en bg es fi fr hi id it ja ko nl no pl pt ru sk sv tr uk zh
# do 
#     echo $lan
#     python mask.py --mask_task mlm  --fdataset wietsedv/udpos28  --shot 128 --seed 42 --device 0 --mask fisher --batch_size 4 --source $lan &
#     wait
# done