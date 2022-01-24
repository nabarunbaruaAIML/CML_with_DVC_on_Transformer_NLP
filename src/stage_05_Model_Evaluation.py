import argparse
import os
import shutil
from tqdm import tqdm
import logging
import json
import numpy as np
from datasets import load_from_disk,load_metric
from src.utils.all_utils import read_yaml,Onnx_Sesion,create_bucket,Upload_to_S3,Weights_Baises,read_json,parameters,create_directory,has_same_value,quantize_onnx_model,CompressModel 
from transformers import AutoTokenizer,AutoConfig,TensorType
from transformers.models.albert import AlbertOnnxConfig
from transformers.onnx.features import FeaturesManager
# from transformers.onnx.convert import validate_model_outputs
from torch.onnx import export
from functools import partial, reduce
from dotenv import load_dotenv
import wandb
from io import BytesIO
from itertools import chain
import torch
import onnx
from onnxruntime.transformers.onnx_model import OnnxModel
from onnxruntime import InferenceSession, SessionOptions
from datasets import Dataset
from datasets import load_dataset
import boto3



"""
This method  reads key-value pairs from a .env file and can set them as environment variables
for example : os.getenv('AWS_SECRET_ACCESS_KEY')
Properties like AWS_SECRET_ACCESS_KEY , AWS_ACCESS_KEY_ID  are set in the environment for example these are set in Github
"""
load_dotenv()

def supported_features_mapping(*supported_features, onnx_config_cls=None):
    """Generates the mapping between supported features and their corresponding OnnxConfig."""
    if onnx_config_cls is None:
        raise ValueError("A OnnxConfig class must be provided")

    mapping = {}
    for feature in supported_features:
        if "-with-past" in feature:
            task = feature.replace("-with-past", "")
            mapping[feature] = partial(onnx_config_cls.with_past, task=task)
        else:
            mapping[feature] = partial(onnx_config_cls.from_model_config, task=feature)

    return mapping

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
# create_directory([log_dir])
os.makedirs(log_dir, exist_ok=True)
"""
The logging configurations are set here like logging level ,file name etc.
"""
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

"""
>> 5 >> This is the Stage where we convert trained Weights to Onnx Format.
"""
STAGE = 'Stage 05'
def main(config_path):
    
    config = read_yaml(config_path.config)
    params = read_yaml(config_path.params)
    
    """AWS S3 bucket Credentials"""
    AWS_ACCESS_KEY_ID= os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY= os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION= os.getenv('AWS_DEFAULT_REGION')
    
    s3 = boto3.resource(
                            service_name='s3',
                            region_name=AWS_DEFAULT_REGION,
                            aws_access_key_id= AWS_ACCESS_KEY_ID,
                            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                        )
     
    
    """Wandb Integration starts here with the API KEY"""
    Wandb_API= os.getenv('WANDB_API_KEY')
    
    wandb.login(key=Wandb_API)
    
    """ Parameter Initialization"""
    TrainingArgument = params['TrainingArgument']
    metric_name = TrainingArgument['metric_name']
    WANDB_PROJECT = TrainingArgument['WANDB_PROJECT']
    model = params['model']
    max_length =  model['max_length']
    
    """ Configuration Initialization"""
    Onnx_output = config['Onnx_Output']
    Model_Repo = config['Model_Repo']
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    Onnx_output_path = os.path.join(artifacts_dir,Onnx_output) 
    best = artifacts['Best_Dir']
    Best_path = os.path.join(artifacts_dir,best) 
    
    """Model Path Initialization"""
    Org_Onnx_model = "model.onnx"
    Original_Onnx_Model = os.path.join(Onnx_output_path,Org_Onnx_model)
    model_compressed ="model_compressed.onnx"
    Compressed_Onnx_output_path = os.path.join(Onnx_output_path,model_compressed)
    model_Quantized = "model_Quantized.onnx"
    Quantized_Onnx_output_path = os.path.join(Onnx_output_path,model_Quantized )
    model_compressed_opt= "model_compressed-opt.onnx"
    Compressed_opt_Onnx_output_path = os.path.join(Onnx_output_path,model_compressed_opt)    
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    Dataset_dir = artifacts['Dataset_dir']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)   
    Dataset_path = os.path.join(DownloadData_path ,Dataset_dir)
    
    #####################################################################
    ######################## Depricated Code ############################
    #####################################################################
    # """ Loading Dataset"""
    # # dataset = load_from_disk(Dataset_path)
    # dataset = load_dataset('csv',  data_files = 'artifacts/Data/atis_intents_test.csv')
    # dataset = dataset['train']
    
    # for i,v in enumerate(dataset.features):
    #     if(i==0):
    #         dataset= dataset.rename_column(v.lower(),'labels')
    #     if(i==1):
    #         dataset= dataset.rename_column(v.lower(),'text')
    #   
    # """Loading Tokenizer"""
    # tokenizer = AutoTokenizer.from_pretrained(Best_path ,cache_dir =  base_model_path, use_fast=use_fast)
    # def tokenize_function(example):
    #     return tokenizer(example["text"], return_tensors="np",truncation=True,padding ='max_length',max_length=128) 
    # logging.info(f"Loaded Tokenizer of Model {model_name} Succefully !")
    # dataset_new = dataset.map(tokenize_function, batched=True,remove_columns= dataset.features)
    # dataset = load_dataset('csv',  data_files = string)
    #####################################################################
    
    
    """Computing Matircs for All the Onnx Models"""
    metric_loaded = load_metric(metric_name)
    dataset = load_from_disk(Dataset_path)
    dataset_new = dataset['test']
    
    Orginal_Label,Orginal_Prediction = Onnx_Sesion(Original_Onnx_Model,dataset_new,max_length)
    Compressed_Label,Compressed_Prediction = Onnx_Sesion(Compressed_Onnx_output_path,dataset_new,max_length)
    Compressed_opt_Label,Compressed_opt_Prediction = Onnx_Sesion(Compressed_opt_Onnx_output_path,dataset_new,max_length)
    Quantized_Label,Quantized_Prediction = Onnx_Sesion(Quantized_Onnx_output_path,dataset_new,max_length)
    
    Orginal_Eval = metric_loaded.compute(predictions=Orginal_Prediction, references=Orginal_Label)
    Compressed_Eval = metric_loaded.compute(predictions=Compressed_Prediction, references=Compressed_Label)
    Compressed_opt_Eval = metric_loaded.compute(predictions=Compressed_opt_Prediction, references=Compressed_opt_Label)
    Quantized_Eval = metric_loaded.compute(predictions=Quantized_Prediction , references=Quantized_Label)
    logging.info(f"Orginal Onnx Model Evalution {Orginal_Eval}")
    logging.info(f"Compressed Onnx Model Evalution {Compressed_Eval}")
    logging.info(f"Compressed Optimized Onnx Model Evalution {Compressed_opt_Eval}")
    logging.info(f"Quantized Onnx Model Evalution {Quantized_Eval}")
    
    
    """Checking which weight to Upload"""
    unwanted_file = []
    unwanted_file.append('.gitignore')
    unwanted_file.append(Org_Onnx_model)
     
    
    if(Compressed_Eval['matthews_correlation']> Compressed_opt_Eval['matthews_correlation']):
        unwanted_file.append(model_compressed_opt)
        if(Compressed_Eval['matthews_correlation']> Quantized_Eval['matthews_correlation']):
            unwanted_file.append(model_Quantized)  
        else:
            unwanted_file.append(model_compressed)                      
    else:
        unwanted_file.append(model_compressed)
        if(Compressed_opt_Eval['matthews_correlation']> Quantized_Eval['matthews_correlation']):
            unwanted_file.append(model_Quantized)  
        else:
            unwanted_file.append(model_compressed_opt)
          
    """S3 Bucket Uploading Files"""    
    if(s3.Bucket(Model_Repo) in s3.buckets.all()):
        logging.info(f"S3 Bucket Exist {Model_Repo}")
        Upload_to_S3(s3,Model_Repo, [Best_path,Onnx_output_path,'logs'],unwanted_file)
        logging.info(f"Files uploaded to S3 Bucket {Model_Repo}")
    else:
        yes = create_bucket(Model_Repo, region= None)
        if yes:
            logging.info(f"S3 Bucket {Model_Repo} Created") 
            Upload_to_S3(s3,Model_Repo, [Best_path,Onnx_output_path,'logs'],unwanted_file)
            logging.info(f"Files uploaded to S3 Bucket {Model_Repo}")
        else:
            logging.info(f"S3 Bucket {Model_Repo} Not Created. Weights will not be Transfer to S3 bucket")
     
    
    """W&B Onnx Model Upload"""
    run = wandb.init(project= WANDB_PROJECT,job_type="upload",name='Onnx')
    Weights_Baises(run,[Onnx_output_path],unwanted_file)
    logging.info(f"Model Uploaded to Weights and Baises Successfully")
    

"""Marks the starting point of Stage >> 3 >>"""
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    # args.add_argument("--secret","-s", default="configs/secrets.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f">>>>> {STAGE} started <<<<<")
        main(parsed_args)
        logging.info(f">>>>> {STAGE}  completed! <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e