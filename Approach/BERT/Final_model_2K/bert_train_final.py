# Import

import re
from os import listdir
from os.path import isfile, join
import pandas as pd

import tensorflow as tf
import time
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import torch
import nltk
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
import os

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
    
#%matplotlib inline

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


start_time = time.time()
# Get all files
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
output_text = ''
_iteration_=0
for i in range(5):
    #train_files = listdir_path("/local/users/ulede/BERT/train_set_synth")
    train_files = listdir_path(".\Sequence-Labeling-for-Reference-Parsing-of-Cyrillic-Script-Scholarly-Data\Synthetic_data\train_set_synth")
    
    sample500 = random.sample(train_files , 2000)
    
    onlyfiles = sample500
    
    df_data = pd.DataFrame({"Text": [], "Labels": []})
    LABELS_TXT = ["title","journal","year","pages","number","volume","author","address","publisher","pagetotal","booktitle"]
    for file in onlyfiles:
        file2 = open(file, encoding='utf-8', mode='r')
        text = file2.read()
        tokens = text.split()
        labels_text = []
        for token in tokens:
            label_ind = "not found"
            for label in LABELS_TXT:
                if re.search("<{}>".format(label),token):
                    labels_text.append(label)
                    label_ind = "found"
            if label_ind == "not found":
                labels_text.append("other")
                
        df_append = pd.DataFrame({'Text' : [re.sub(string=text,pattern = "<.*?>", repl="")] , 'Labels':[' '.join(labels_text)]})
        df_data = df_data.append(df_append , ignore_index = True)
    
    print("Data was loaded")
    
    
    
    LABELS_TXT = ["other","title","journal","year","pages","number","volume","author","address","publisher","pagetotal","booktitle"]
    tag_values = list(set(LABELS_TXT))
    tag_values
    tag_values.append("PAD")
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    print(tag2idx)
    
    sentences = df_data["Text"]
    labels = df_data["Labels"]
    
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(sent, labs) for sent, labs in zip(sentences, labels)]
    
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    
    MAX_LEN = 512
    bs = 8
    
    print("TEST GPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(torch.cuda.get_device_name(0))
    print("TEST DONE")
    
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, 
                              dtype="long",
                              value=0.0,
                              truncating="post", padding="post")
    
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")
                         
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    
    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                                random_state=2018, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=2018, test_size=0.1)
    
    tr_inputs = torch.LongTensor(tr_inputs)
    val_inputs = torch.LongTensor(val_inputs)
    tr_tags = torch.LongTensor(tr_tags)
    val_tags = torch.LongTensor(val_tags)
    tr_masks = torch.LongTensor(tr_masks)
    val_masks = torch.LongTensor(val_masks)
    
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)
    
    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
    
    print(transformers.__version__)
    
    print("load bert model")
    model = BertForTokenClassification.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda();
    print("bert model loaded!")
    
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=3e-5,
        eps=1e-8
    )
    
    from transformers import get_linear_schedule_with_warmup
    
    epochs = 3
    max_grad_norm = 1.0
    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    print("START TRAINING")
    
    ## Store the average loss after each epoch so we can plot them.
    loss_values, validation_loss_values = [], []
    
    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.
    
        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0
    
        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
    #        batch = tuple(t for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
    
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))
    
        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
    
    
        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
    
        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions , true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
    #        batch = tuple(t for t in batch)
    
            b_input_ids, b_input_mask, b_labels = batch
    
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)
    
        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                     for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        valid_tags = [tag_values[l_i] for l in true_labels
                                      for l_i in l if tag_values[l_i] != "PAD"]
        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags, average = "weighted")))
        print()
        
    print("DONE TRAINING")
    
    
    print("try to save model")

    torch.save(model, f"/local/users/ulede/BERT/final_model/model_{str(_iteration_)}.pth")
    
    print("MODEL SAVED!")
    
    print("tag_values order:")
    output_text=""
    for i in range(len(tag_values)):
      print(str(i)+ f":{tag_values[i]}  \n")
      output_text+= str(i)+ f":{tag_values[i]}  \n"
        
    myfile = open("/local/users/ulede/BERT/final_model/" + f"tag_values_{str(_iteration_)}", "w",encoding="utf-8")
    myfile.write(output_text)
    myfile.close()
    
    path_test = "/local/users/ulede/BERT/real_text4/"
    onlyfiles2 = [f for f in listdir(path_test) if isfile(join(path_test, f))]
    
    all_ref_test = []
    for _file_ in onlyfiles2:
        file2 = open(path_test + _file_, encoding='utf-8', mode='r')
        text = file2.read()
        text = text.split("\n\n")
        all_ref_test+=text
        
    #print(len(all_ref_test))
    
    df_data = pd.DataFrame({"Text": [], "Labels": []})
    
    for ref in all_ref_test:
        tokens = []
        labels = []
    
        for f in ref.split("<"):
            if f == "":
                pass
            else:
                text = f.split(">")
                if len(text)>1:
                    text_token = test_tokenize(text[1])
    
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
    
    real_label_eval=[]
    new_label_eval=[]
    #print(df_data)
    test_text = df_data.Text
    test_label = df_data.Labels
    
    
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
    
              new_tokens.append(token)
              
      token_label = tokenize_and_preserve_labels(test,test_labels)     
      real_tokens, real_labels = [], []
      for token, label_idx in zip(token_label[0], token_label[1]):
          if token.startswith("##"):
              real_tokens[-1] = real_tokens[-1] + token[2:]
          else:
              real_labels.append(label_idx)
    
              real_tokens.append(token)
      real_label_eval+= real_labels
      new_label_eval+= new_labels[1:-1]
    
    
      
    real_label_eval_num, new_label_eval_num = [],[]
    for r,n in zip(real_label_eval,new_label_eval):
      real_label_eval_num.append(tag2idx[r])
      new_label_eval_num.append(tag2idx[n])
    
    target_names = ['author','publisher','volume','other','number','pages','address','journal','title','year']
    target_names = list(tag2idx.keys())
    target_names.remove("PAD")
    target_names.remove("pagetotal")
    target_names.remove("booktitle")
    print(classification_report(real_label_eval_num,new_label_eval_num,target_names= target_names,digits=3))
    
    output_text += f"Iteration No. {_iteration_} \n\nEVALUATION ON REAL CITATION STRING DATA SET. 711 REFERENCES\n\n\n"
    output_text += "f1 score: " + str(f1_score(real_label_eval,new_label_eval,average="micro"))+"\n" + "accuracy_score: " + str(accuracy_score(real_label_eval,new_label_eval))+"\n"+"recall_score: " + str(recall_score(real_label_eval,new_label_eval,average="micro"))+"\n"+"precision_score: " + str(precision_score(real_label_eval,new_label_eval,average="micro"))+"\n"+"f1 score all classes: " + str(f1_score(real_label_eval,new_label_eval,average=None))+ "\n\n"+str(classification_report(real_label_eval_num,new_label_eval_num,target_names= target_names,digits=3))
    _iteration_+=1




myfile = open("/local/users/ulede/BERT/final_model/" + "eval_2K.txt", "w",encoding="utf-8")
myfile.write(output_text)
myfile.close()

end_time = time.time()
time_elapsed = (end_time - start_time)
print("Dauer in min: "+ str(time_elapsed/60))

