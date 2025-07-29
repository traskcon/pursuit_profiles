"""
SOURCE: https://github.com/mpchang/uncovering-missed-tackle-opportunities/blob/main/code/load_data.ipynb 

This script is used to generate and save analysis-ready data from raw data:

1) Load raw data
2) Clean and standardize data
3) Preprocess data into features and targets
4) Save data for future use

"""

import numpy as np
import pandas as pd
import torch
import random
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

from data_preprocessing import aggregate_data, annotate_missed_tackle_frames, annotate_tackle_frames, build_training_tackle_sequences
from data_cleaning import clean_data, optimize_memory_usage
from constants import TACKLENET_FEATURE_DICT

VERBOSE = False
root_dir = os.getcwd()

# set manual custom seed for reproducibility
def set_random_seed(value): 
    g = torch.manual_seed(value)   
    np.random.seed(value)
    random.seed(value)
    torch.backends.cudnn.deterministic=True
    return g

plays_fname = os.path.join(root_dir, "data/plays.csv")
players_fname = os.path.join(root_dir, "data/players.csv")
tackles_fname = os.path.join(root_dir, "data/tackles.csv")
tracking_fname_list_train = [os.path.join(root_dir, f"data/tracking_week_{i}.csv") for i in range(1,9)]
tracking_fname_list_test = [os.path.join(root_dir, "data/tracking_week_9.csv")]

# variables to store train vs. test data for later 
tackle_sequences_train = []
tackle_sequence_test = []
df_tracking_train = []
df_tracking_test = []
df_tackles_train = []
df_tackles_test = []
means = None
stds = None
unnorm_tensor_stack_train = None
norm_tensor_stack_train = None
unnorm_tensor_stack_test = None
norm_tensor_stack_test = None