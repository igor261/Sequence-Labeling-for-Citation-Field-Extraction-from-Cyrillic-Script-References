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

apa = listdir_path("/local/users/ulede/BERT/labelled_text/apa_fine_grained_clean")
gost2003 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2003_fine_grained_clean")
gost2008 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2008_fine_grained_clean")
gost2006 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2006_fine_grained_clean")

files = apa + gost2003 + gost2008 + gost2006


# 2K citation strings in test set with fixed distribution of citation styles
apa_part = int(len(apa)/len(files)*2000)+1
gost2003_part = int(len(gost2003)/len(files)*2000)+1
gost2008_part = int(len(gost2008)/len(files)*2000)+1
gost2006_part = int(len(gost2006)/len(files)*2000)+1

number_papers = [apa_part,gost2003_part,gost2006_part,gost2008_part]
ciation_style = ["apa","gost2003","gost2006","gost2008"]
for n,s in zip(number_papers,ciation_style):
    print(f"{s} style has: {n} Citation Strings")

random.seed(10)
files_style = [apa,gost2003,gost2006,gost2008]
test_set = []
for n,f in zip(number_papers,files_style): 
    test_set+=(random.sample(f , n))

train_set=[]
for f in files:
    if f in test_set:
        pass
    else:
        train_set.append(f)

test_set_dir = "/local/users/ulede/BERT/test_set_synth/"

for f in test_set:
    original = f 
    target = os.path.join(test_set_dir,f.split("/")[7].replace(".xml",f.split("/")[6]+".xml"))
  
    shutil.copyfile(original, target)
    
train_set_dir = "/local/users/ulede/BERT/train_set_synth/"

for f in train_set:
    original = f 
    target = os.path.join(train_set_dir,f.split("/")[7].replace(".xml",f.split("/")[6]+".xml"))
  
    shutil.copyfile(original, target)

print(f"The test set has a size of: {len(listdir_path(test_set_dir))} citation strings")
print(f"The train set has a size of: {len(listdir_path(train_set_dir))} citation strings")
