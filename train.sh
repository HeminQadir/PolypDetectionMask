#!/usr/bin/env bash

#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0 #,1,2,3

for i in 10 #20 30 40 50 60 
do
    #python3 all_in_one.py --device-ids 0 --limit 10000 --batch-size 2 --n-epochs $i --fold 1 --root 'runs/MDeNetplus'
    python3 train.py --device-ids 0 --limit 10000 --batch-size 2 --n-epochs $i --fold 0 --root 'runs/AlbuNet34' --model AlbuNet34

done



# Use GAN with AlbuNet34