#!/bin/bash

#parallel -j 8 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size={2} --kernal_size={3} ::: 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.006 0.0075 0.01 ::: 32 64 128 ::: 2 3 4 5 6 7 8 9 10

#parallel -j 8 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size=32 --conv1_size=16 --conv2_size=16 --fc1_size=24::: 0.002 0.0025 0.004

#parallel -j 10 python src/main.py --learning_rate=0.00286207 --epoch_count=100 --batch_size=32 --conv1_size={1} --conv2_size={2} --fc1_size={3} ::: 8 12 16 20 24 30 ::: 2 4 12 16 ::: 10 12 16 20 24 30


# parallel ::: \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=8 --conv2_size=2 --fc1_size=22" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=10 --conv2_size=4 --fc1_size=10" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=12 --conv2_size=2 --fc1_size=18" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=12 --conv2_size=4 --fc1_size=20" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=16 --conv2_size=4 --fc1_size=10" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=20 --conv2_size=4 --fc1_size=8" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=24 --conv2_size=2 --fc1_size=8" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=28 --conv2_size=4 --fc1_size=18" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=28 --conv2_size=4 --fc1_size=30" \
#   "python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size=32 --conv1_size=30 --conv2_size=2 --fc1_size=16"

#python src/main.py

#parallel -j 10 python src/main.py --learning_rate=0.001 --epoch_count=50 --batch_size=64 --conv1_size={1} --conv2_size={2} --fc1_size={3} --kernal_size1=4 --kernal_size2=4 ::: 8 12 16 20 24 30  :::  2 4 12 16 ::: 8 10 12 16 20 24 30

#parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=80 --batch_size={2} --conv1_size=16 --conv2_size=16 --fc1_size=16 --kernal_size1=4 --kernal_size2=4 ::: 0.001 0.002 0.01 0.02 ::: 8 16 32

#parallel -j 10 python src/main.py --learning_rate=0.002 --epoch_count=100 --batch_size={1} --dropout1={2}  --dropout2={3}  --dropout3={4} ::: 32 64 256 512 1024 ::: 0.1 0.2 0.3 0.4 0.5 0.8 ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.8 ::: 0.1 0.2 0.3 0.4 0.5 0.8 