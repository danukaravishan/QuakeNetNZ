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


#parallel -j 10 python src/main.py --learning_rate=0.002 --epoch_count=200 --batch_size=1024 --conv1_size=32 --conv2_size=32 --conv3_size=32 --fc1_size=44 --fc2_size=18 --l2_decay={1} --dropout1={2} --dropout2={3} ::: 5e-4 5e-3 8e-4 8e-3 3e-3 2e-3 2e-4 8e-5 2e-5 2e-3 ::: 0.1 0.2 0.3 0.4 0.5  ::: 0.1 0.2 0.3 0.4 0.5

#parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=200 --batch_size=1024 --conv1_size={1} --conv2_size={2} --conv3_size={3} --fc1_size={4} --fc2_size={5} --l2_decay=0.0008 --dropout1=0.3 --dropout2=0.2 ::: 8 16 24 32 44 64 ::: 8 16 32 44 64 ::: 8 16 32 64 88 ::: 16 32 44 64 88 ::: 8 16 32 44 64
#parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=200 --batch_size=1024 --conv1_size=32 --conv2_size=32 --conv3_size=32 --fc1_size=44 --fc2_size=18 --l2_decay=0.0008 --dropout1=0.3 --dropout2=0.2 ::: 0.0001 0.0002 0.0003 0.0004 0.0005 0.0006 0.0007 0.0008 0.0009 0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 

#python src/hyperparam_opt.py 

#python src/main.py --learning_rate=0.00021683919249716394 --epoch_count=200 --batch_size=256 --conv1_size=24 --conv2_size=28 --conv3_size=24 --fc1_size=44 --fc2_size=40 --l2_decay=0.0003717615671854365 --dropout1=0.3191923277183404 --dropout2=0.4750640144857251 --kernal_size1=10 --kernal_size2=4 --kernal_size3=5 --model_note="With best hyperparameters"

#parallel -j 10 python src/main.py --learning_rate=0.0002 --epoch_count=100 --batch_size=1024 --conv1_size=32 --conv2_size=32 --conv3_size=32 --fc1_size=44 --fc2_size=18 --l2_decay=0.0008 --dropout1=0.3 --dropout2=0.2 --kernal_size1=4 --kernal_size2=4 --kernal_size3=4 --wavelet_name={1} --wavelet_level={2} ::: "db2" "db4" "db8" "db20" "coif2" "coif4" "coif8" "coif20" "haar" "bior1.3" ::: 2 3 4 6 8 10

#parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size={2} --l2_decay={3} ::: 0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 ::: 32 64 256 512 1024 ::: 0.00001 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01

parallel -j 10 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size=64 --l2_decay=0.001 ::: 0.00001 0.00002 0.00005 0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 