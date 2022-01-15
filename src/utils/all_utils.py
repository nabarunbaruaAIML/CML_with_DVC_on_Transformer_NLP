import pandas as pd
import numpy as np
#from transformers import AutoTokenizer

import yaml
import os
import json
import logging
import shutil
import time
import boto3
from datasets import load_dataset
from datasets import Dataset
from datasets import ClassLabel


def read_json(path_to_json: str) -> dict:
    with open(path_to_json) as json_file:
        data = json.load(json_file)
        logging.info(f"JSON file: {path_to_json} loaded successfully")
    return data


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content

def create_directory(dirs: list):
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"directory is created at {dir_path}")


def save_local_df(data, data_path, index_status=False):
    data.to_csv(data_path, index=index_status)
    logging.info(f"data is saved at {data_path}")


def save_reports(report: dict, report_path: str, indentation=4):
    with open(report_path, "w") as f:
        json.dump(report, f, indent=indentation)
    logging.info(f"reports are saved at {report_path}")
    
def copy_file(source_download_dir, local_data_dir):
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)
    
    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)
        logging.info(f"Copying File from {src} to {dest} Completed! Succefully")
    # for file in tqdm(list_of_files, total=N, desc=f"copying file from {source_download_dir} to {local_data_dir}", colour="green"):
    #     src = os.path.join(source_download_dir, file)
    #     dest = os.path.join(local_data_dir, file)
    #     shutil.copy(src, dest)
def copy_file_csv(source_download_dir, local_data_dir):
    list_of_files =[ listfile for listfile in os.listdir(source_download_dir) if listfile.endswith('.csv') ] 
    N = len(list_of_files)
    
    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        shutil.copy(src, dest)
        logging.info(f"Copying File from {src} to {dest} Completed! Succefully")
    # for file in tqdm(list_of_files, total=N, desc=f"copying file from {source_download_dir} to {local_data_dir}", colour="green"):
    #     src = os.path.join(source_download_dir, file)
    #     dest = os.path.join(local_data_dir, file)
    #     shutil.copy(src, dest)
def copy_file_from_S3(s3,S3_location, local_data_dir):
    # print('S3 load start')    
    obj = s3.Bucket(S3_location)
    list_of_files =[ i.key for i in obj.objects.all() ] 
    N = len(list_of_files)
    
    for file in list_of_files:
        
        # src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)
        
        # print(f'Destination{dest}')
        
        # shutil.copy(src, dest)
        s3.Bucket(S3_location).download_file(Key=file, Filename=dest)
        logging.info(f"Copying File from S3 to {dest} Completed! Succefully")
        
        # print(f"Copying File from S3 to {dest} Completed! Succefully")
    
def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_input_ids(df,tokeniser,padding,max_length,truncation,stop_words=None):
    df = df.tolist()
    output = tokeniser(text=df , truncation=truncation,padding =padding,max_length=max_length)
    return output
    # inp_ids = []
    # attension_mask= []
    # for i in df:
    #     output = tokeniser(text=i, truncation=truncation,padding =padding,max_length=max_length)#,return_tensors='np' )
    #     inp_ids.append(output['input_ids'])
    #     attension_mask.append(output['attention_mask'])
    # return inp_ids,attension_mask

def read_data(string,tokenizer,padding,max_length,truncation,stop_words=None):
    # json = {}
    df = pd.read_csv(string)
    DataFrame_Filtered = df.iloc[:,0:2]
    Label_set = set(DataFrame_Filtered.iloc[:,0])
    Label_set = sorted(Label_set)
    label_num = len( Label_set )
    id2label = {k:v for k,v in enumerate(Label_set,0)}
    label2id = {v:k for k,v in enumerate(Label_set,0)}
    # inp_ids,attension_mask = get_input_ids(DataFrame_Filtered.iloc[:,1],tokenizer,padding,max_length,truncation,stop_words)
    json = get_input_ids(DataFrame_Filtered.iloc[:,1],tokenizer,padding,max_length,truncation,stop_words)
    DataFrame_Filtered1 = DataFrame_Filtered.iloc[:,0].map(lambda x: label2id[x]).copy()
    Label = DataFrame_Filtered1.tolist()
    # json['input_ids'] =inp_ids
    # json['attention_mask']= attension_mask
    json['labels']= Label 
    logging.info(f"Read Completed and Tokenise Succefully")
    return json,id2label,label2id,label_num,Label_set

def read_dataset(string,tokenizer,padding,max_length,truncation ):
    # json = {}
    df = pd.read_csv(string)
    # DataFrame_Filtered = df.iloc[:,0:2]
    Label_set = set(df.iloc[:,0])
    Label_set = sorted(Label_set)
    label2id = {v:k for k,v in enumerate(Label_set,0)}
    id2label = {k:v for k,v in enumerate(Label_set,0)}
    label_num = len( Label_set )
    del df
    dataset = load_dataset('csv',  data_files = string)
    dataset = dataset['train']
    for i,v in enumerate(dataset.features):
        if(i==0):
            dataset= dataset.rename_column(v.lower(),'labels')
        if(i==1):
            dataset= dataset.rename_column(v.lower(),'text')
    
    def labelMap(example):
        # example['labels']=label2id[example['labels']]
        return {'labels':[label2id[i] for i in example['labels'] ] }#example
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=truncation,padding =padding,max_length=max_length)
    dataset = dataset.map(labelMap, batched=True)
    VClassLabel = ClassLabel(names=Label_set)
    dataset =dataset.cast_column('labels',VClassLabel)
    dataset = dataset.align_labels_with_mapping(label2id, "labels")
    dataset = dataset.map(tokenize_function, batched=True)
    # dataset = dataset_new.train_test_split(test_size=0.1)
    
    # This Column could be kept as it is and Trainer & Model would be have takecare, it's been remove to save memory
    dataset = dataset.remove_columns(['text'])  
    # dataset.set_format('torch')
    logging.info(f"Read Completed and Tokenise Succefully")
    
    return dataset,id2label,label2id,label_num,Label_set

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")