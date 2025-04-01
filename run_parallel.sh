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

#parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=50 --batch_size=32 --conv1_size=8 --conv2_size=2 --fc1_size=22 --kernal_size1={2} --kernal_size2={3} ::: 0.002 0.00286207 ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

#python src/main.py --learning_rate=0.00286207 --epoch_count=150 --batch_size=32 --conv1_size=8 --conv2_size=2 --fc1_size=22


parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size={2}  ::: 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1  ::: 32 64 128 256