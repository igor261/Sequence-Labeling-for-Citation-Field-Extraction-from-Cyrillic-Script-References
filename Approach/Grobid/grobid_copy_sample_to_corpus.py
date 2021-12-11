import pandas as pd
import os
import numpy as np

import re
# import json
import matplotlib.pyplot as plt

import random
import shutil
from collections import Counter

def listdir_path(d):
    # Return full path of all files & directories in directory
    list_full_path = []
    for path in os.listdir(d):
        full_path = os.path.join(d, path)
        list_full_path.append(full_path)
    return list_full_path
    
#path_input = "/local/users/ulede/grobid_citation_token_level"
path_input = ".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Grobid\grobid_training_data"

path_output = "./grobid-0.6.1/grobid-trainer/resources/dataset/citation/corpus/"

random.seed(10)

files = listdir_path(path_input)

sample = random.sample(files, 2000)
print(len(sample))

for f in sample:
  original = f 
  target = os.path.join(path_output,f.split("/")[-1])

  shutil.copyfile(original, target)