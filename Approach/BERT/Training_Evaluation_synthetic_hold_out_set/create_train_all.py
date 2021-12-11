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

#apa = listdir_path("/local/users/ulede/BERT/labelled_text/apa_fine_grained_clean")
#gost2003 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2003_fine_grained_clean")
#gost2008 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2008_fine_grained_clean")
#gost2006 = listdir_path("/local/users/ulede/BERT/labelled_text/gost2006_fine_grained_clean")

#files = apa + gost2003 + gost2008 + gost2006

files = listdir_path("/local/users/ulede/BERT/train_set_synth")

random.seed(10)

sample500 = random.sample(files , 500)
sample1k = random.sample(files , 1000)
sample2k = random.sample(files , 2000)
sample3k = random.sample(files , 3000)
sample5k = random.sample(files , 5000)
sample10k = random.sample(files , 10000)
sample20k = random.sample(files , 20000)
sample50k = random.sample(files , 50000)
sample100k = random.sample(files , 100000)

for i in [sample500,sample1k,sample2k,sample3k,sample5k,sample10k,sample20k,sample50k,sample100k]:
  len_k = int(len(i)/1000)
  out_dir = os.path.join("/local/users/ulede/BERT/labelled_text/","train_"+str(len_k)+"K")
  #styles = []
  #for s in i:
  #    styles.append(s.split("/")[5])
           
  #print(Counter(styles))
  "/local/users/ulede/BERT/labelled_text/train_0K/"

  for f in i:
      original = f 
      target = os.path.join(out_dir,f.split("/")[-1])
  
      shutil.copyfile(original, target)