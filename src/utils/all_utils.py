import pandas as pd
import numpy as np
#from transformers import AutoTokenizer

import yaml
import os
import json
import logging
import shutil
import time

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
    
def get_timestamp(name):
    timestamp = time.asctime().replace(" ", "_").replace(":", "_")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name

def get_input_ids(df,tokeniser,padding,max_length,truncation,stop_words=None):
    inp_ids = []
    attension_mask= []
    df = df.tolist()
    for i in df:
        output = tokeniser(i, truncation=truncation,padding =padding,max_length=max_length)
        inp_ids.append(output['input_ids'])
        attension_mask.append(output['attention_mask'])
    return inp_ids,attension_mask

def read_data(string,tokenizer,padding,max_length,truncation,stop_words=None):
    json = {}
    df = pd.read_csv(string)
    DataFrame_Filtered = df.iloc[:,0:2]
    Label_set = set(DataFrame_Filtered.iloc[:,0])
    Label_set = sorted(Label_set)
    label_num = len( Label_set )
    id2label = {k:v for k,v in enumerate(Label_set,0)}
    label2id = {v:k for k,v in enumerate(Label_set,0)}
    inp_ids,attension_mask = get_input_ids(DataFrame_Filtered.iloc[:,1],tokenizer,padding,max_length,truncation,stop_words)
    DataFrame_Filtered1 = DataFrame_Filtered.iloc[:,0].map(lambda x: label2id[x]).copy()
    Label = DataFrame_Filtered1.tolist()
    json['input_ids'] =inp_ids
    json['attention_mask']= attension_mask
    json['labels']= Label 
    logging.info(f"Read Completed and Tokenise Succefully")
    return json,id2label,label2id,label_num,Label_set

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")