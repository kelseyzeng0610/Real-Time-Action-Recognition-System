import mpose 
import tensorflow as tf
import numpy as np

from utils.trainer import Trainer
from utils.tools import read_yaml, Logger
#load the model





seq_len = 30

input_feature = 13 * 4


# Load configuration if needed
config = read_yaml('utils/config.yaml') 


split = config.get('SPLIT', 1)  # Adjust as per your config
fold = config.get('FOLD', 0)    # Adjust as per your config
trainer = Trainer(config, logger = None, split=split, fold=fold)
# # Initialize your model

# data = mpose.MPOSE()

trainer.get_data()
trainer.get_model()

# # Load pretrained weights
trainer.fine_tune(pretrained_weights_path='bin/AcT_posenet_large.h5', fine_tune_epochs=5)
test_loss, test_accuracy = trainer.evaluate()

print(f"Fine-tuned model test accuracy: {test_accuracy * 100:.2f}%")






# Compile the model
