from utils import *
from config import Config, MODE_TYPE
from train import train
from test import test
from extract_window_db import extract_data, extract_stead_data, extract_p_data_for_new_data
from database_op import split_data, split_data_randomly

def main():
   cfg = Config()

   if cfg.MODE == MODE_TYPE.IDLE:
      print()

   elif cfg.MODE == MODE_TYPE.EXTRACT_DATA:
      #extract_data()
      extract_p_data_for_new_data(cfg)
      split_data_randomly()
      #extract_stead_data()

   elif cfg.MODE == MODE_TYPE.SPLIT_DATA:
      #split_data()
      split_data_randomly()

   elif cfg.MODE == MODE_TYPE.TRAIN:
      train(cfg)

   elif cfg.MODE == MODE_TYPE.PREDICT:
      test(cfg)

   elif cfg.MODE == MODE_TYPE.ALL:
      train(cfg)
      test(cfg)

   elif cfg.MODE == MODE_TYPE.HYPER_PARAM:
      hyper_param_opt()

if __name__ == "__main__":
    main()