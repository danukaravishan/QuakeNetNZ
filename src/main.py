from utils import *
from config import Config, MODE_TYPE
from train import train
from test import test
from extract_window_db import extract_data
from database_op import split_data

def main():
   cfg = Config()

   if cfg.MODE == MODE_TYPE.IDLE:
      print()

   elif cfg.MODE == MODE_TYPE.EXTRACT_DATA:
      extract_data()

   elif cfg.MODE == MODE_TYPE.SPLIT_DATA:
      split_data()

   elif cfg.MODE == MODE_TYPE.TRAIN:
      train(cfg)
      test(cfg)

   elif cfg.MODE == MODE_TYPE.PREDICT:
      test(cfg)

   elif cfg.MODE == MODE_TYPE.MODEL_ACCURACY:
      model = PWaveCNN(cfg.BASE_SAMPLING_RATE*cfg.TRAINING_WINDOW)  # Make sure to use the same window_size used during training
      model.load_state_dict(torch.load(cfg.MODEL_FILE_NAME))

if __name__ == "__main__":
    main()