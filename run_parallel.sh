#!/bin/bash

#parallel -j 5 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size={2} ::: 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 0.006 0.0075 0.01 ::: 32 64 128 256
parallel -j 5 python src/main.py --learning_rate={1} --epoch_count=100 --batch_size={2} ::: 0.001 0.0015 0.002 0.0025 0.003 0.004 0.005 ::: 32 64 128
