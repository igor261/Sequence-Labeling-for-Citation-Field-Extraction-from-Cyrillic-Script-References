# Import

import re
from os import listdir
from os.path import isfile, join
import pandas as pd
import os

import tensorflow as tf
import nltk
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import transformers
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification, AdamW
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report

def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence.split(), text_labels.split()):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
    
def test_tokenize(text):
    nltk_tokens = []
    pattern = re.compile("[0-9]+\.")
    pattern2= re.compile(":[0-9]+")
    pattern3= re.compile("([A-Z]\.)|([\w?-?]\.)")
    pattern4 = re.compile(".*\.")

    for t in nltk.word_tokenize(text,preserve_line=True):
        if pattern2.match(t)!=None:
            nltk_tokens.append(":")
            if "." in t:
                nltk_tokens.append(t.replace(":","").replace(".",""))
            else:
                nltk_tokens.append(t.replace(":",""))
            nltk_tokens.append(".")
        elif ((pattern.match(t)!= None) or (pattern3.match(t)== None and pattern4.match(t)!=None)) and t !=".":
            nltk_tokens.append(t.replace(".",""))
            nltk_tokens.append(".")

        else:
            nltk_tokens.append(t)
    return(nltk_tokens)  
     


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

tag2idx = {'author': 0,#
 'publisher': 1,
 'volume': 2,
 'other': 3,#
 'number': 4,
 'pages': 5,
 'address': 6,
 'journal': 7,#
 'title': 8,#
 'year': 9,#
 'PAD': 10}


#tag_values = open("/local/users/ulede/BERT/final_model/tag_values_2","r",encoding="utf-8")
tag_values = open(".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model\tag_values_2","r",encoding="utf-8")
tag_values = tag_values.read()
tag_values = re.sub("\d+\:","",tag_values).replace(" ","").split("\n")[:-1]
#tag_values = ["publisher","title","journal","author","volume","other","pages","year","number","address","PAD"] 
#model = torch.load("/local/users/ulede/BERT/final_model/model_2.pth")
model = torch.load(".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model\model_2.pth")

#path_in = "/local/users/ulede/BERT/real_data_language/"
path_in = ".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Real_annotated_data\labelled_text_per_language"

languages = listdir(path_in)
for lang_dir in languages:
    onlyfiles = [f for f in listdir(os.path.join(path_in,lang_dir)) if isfile(join(os.path.join(path_in,lang_dir), f))]
    path_test = os.path.join(path_in,lang_dir)
    all_ref_test = []
    for _file_ in onlyfiles:
        file2 = open(os.path.join(path_test, _file_), encoding='utf-8', mode='r')
        text = file2.read()
        text = text.split("\n\n")
        all_ref_test+=text
        
    print(len(all_ref_test))
    
    df_data = pd.DataFrame({"Text": [], "Labels": []})
    
    for ref in all_ref_test:
        tokens = []
        labels = []
    #     print(ref)
        for f in ref.split("<"):
            if f == "":
                pass
            else:
                text = f.split(">")
                if len(text)>1:
                    text_token = test_tokenize(text[1])
            #         new_text = " ".join(test_tokenize(text[1]))
            #         tokens+=test_tokenize(text[1])
                    if "/" in text[0]:
                        for i in text_token:
                            tokens.append(i)
                            labels.append("other")
                    else:
                        label = text[0]
                        for i in text_token:
                            tokens.append(i)
                            if label=="author" and i==",":
                                labels.append("other")
                            else:
                                labels.append(label)
                else:
                    text_token = test_tokenize(text[0])
                    for i in text_token:
                        tokens.append(i)
                        labels.append("other")
        new_text = ' '.join(tokens)
        new_labels = ' '.join(labels)
    
            
        df_append = pd.DataFrame({'Text' : [new_text] , 'Labels':[new_labels]})
        df_data = df_data.append(df_append , ignore_index = True)    
    #test_list = open("/local/users/ulede/BERT/real_text/145231899.txt","r",encoding="utf-8")
    #test_list = test_list.read()
    #test_list = test_list.split("\n\n")
    real_label_eval=[]
    new_label_eval=[]
    print(df_data)
    test_text = df_data.Text
    test_label = df_data.Labels
    
    test = df_data.Text[0]
    real_labels = df_data.Labels[0].split(" ")
    
    for test, test_labels in zip(test_text, test_label):
      for values in ["publisher","title","journal","author","volume","other","pages","year","number","address"]:
        test = test.replace(f"<{values}>","").replace(f"</{values}>","")
      test_sentence = test
      tokenized_sentence = tokenizer.encode(test_sentence)
      
      input_ids = torch.LongTensor([tokenized_sentence]).cuda()
      with torch.no_grad():
          output = model(input_ids)
      label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
      tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
      
      new_tokens, new_labels = [], []
      for token, label_idx in zip(tokens, label_indices[0]):
          if token.startswith("##"):
              new_tokens[-1] = new_tokens[-1] + token[2:]
          else:
              label_ = tag_values[label_idx]
              if label_ == "pagetotal":
                    label_ = "pages"
              elif label_ == "booktitle":
                    label_ = "journal"
              
              new_labels.append(label_)
             # new_labels.append(label_idx)
              new_tokens.append(token)
              
      token_label = tokenize_and_preserve_labels(test,test_labels)     
      real_tokens, real_labels = [], []
      for token, label_idx in zip(token_label[0], token_label[1]):
          if token.startswith("##"):
              real_tokens[-1] = real_tokens[-1] + token[2:]
          else:
              real_labels.append(label_idx)
             # new_labels.append(label_idx)
              real_tokens.append(token)
      real_label_eval+= real_labels
      new_label_eval+= new_labels[1:-1]
    
    
      
    real_label_eval_num, new_label_eval_num = [],[]
    for r,n in zip(real_label_eval,new_label_eval):
      real_label_eval_num.append(tag2idx[r])
      new_label_eval_num.append(tag2idx[n])
    
    target_names = ['author','publisher','volume','other','number','pages','address','journal','title','year']

    print(classification_report(real_label_eval_num,new_label_eval_num,target_names= target_names,digits=3))

    output_text = "f1 score: " + str(f1_score(real_label_eval,new_label_eval,average="micro"))+"\n" + "accuracy_score: " + str(accuracy_score(real_label_eval,new_label_eval))+"\n"+"recall_score: " + str(recall_score(real_label_eval,new_label_eval,average="micro"))+"\n"+"precision_score: " + str(precision_score(real_label_eval,new_label_eval,average="micro"))+"\n"+"f1 score all classes: " + str(f1_score(real_label_eval,new_label_eval,average=None))+ "\n\n"+str(classification_report(real_label_eval_num,new_label_eval_num,target_names= target_names,digits=3))
    myfile = open("/local/users/ulede/BERT/final_model/" + f"eval_{lang_dir}.txt", "w",encoding="utf-8")#".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\BERT\BERT_final_model\.."
    myfile.write(output_text)
    myfile.close()