import argparse
import os
import shutil
from tqdm import tqdm
import logging
import json
import numpy as np
from datasets import load_from_disk,load_metric
from src.utils.all_utils import read_yaml,read_json,parameters,create_directory,has_same_value,quantize_onnx_model,CompressModel
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
    """Wandb Integration starts here with the API KEY"""
    Wandb_API= os.getenv('WANDB_API_KEY')
    
    wandb.login(key=Wandb_API)
    
    """ Parameter Initialization"""
    model = params['model']
    model_name = model['base_model']
    use_fast =  model['use_fast']
        
    Onnx = params['Onnx']
    features = Onnx['feature']
    opset = Onnx['opset']
    atol = Onnx['atol']
    
    """ Configuration Initialization"""
    Onnx_output = config['Onnx_Output']
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    best = artifacts['Best_Dir']
    Onnx_output_path = os.path.join(artifacts_dir,Onnx_output) 
    output = os.path.join(Onnx_output_path,"model.onnx")
    Compressed_Onnx_output_path = os.path.join(Onnx_output_path,"model_compressed.onnx")
    Quantized_Onnx_output_path = os.path.join(Onnx_output_path,"model_Quantized.onnx")
    Compressed_opt_Onnx_output_path = os.path.join(Onnx_output_path,"model_compressed-opt.onnx")
    Best_path = os.path.join(artifacts_dir,best) 
    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_path = os.path.join(artifacts_dir,base_model_dir )      
    
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    Dataset_dir = artifacts['Dataset_dir']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)   
    Dataset_path = os.path.join(DownloadData_path ,Dataset_dir)
    
    """ Loading Dataset"""
    # dataset = load_from_disk(Dataset_path)
    dataset = load_dataset('csv',  data_files = 'artifacts/Data/atis_intents_test.csv')
    dataset = dataset['train']
    
    for i,v in enumerate(dataset.features):
        if(i==0):
            dataset= dataset.rename_column(v.lower(),'labels')
        if(i==1):
            dataset= dataset.rename_column(v.lower(),'text')
    def tokenize_function(example):
        return tokenizer(example["text"], return_tensors="np")   
    """Loading Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(Best_path ,cache_dir =  base_model_path, use_fast=use_fast)
    logging.info(f"Loaded Tokenizer of Model {model_name} Succefully !")
    dataset1 = dataset.map(tokenize_function, batched=True,remove_columns= dataset.features)
    # tok_data = tokenizer(dataset['text'], return_tensors="np")
    # print(dataset1[0])
   
    options = SessionOptions()
    options.optimized_model_filepath = output
    session = InferenceSession(output, options, providers=['CPUExecutionProvider'])
    for i,single in enumerate(dataset1):
        # single_tok = single.map(tokenize_function, batched=True,remove_columns= single.features)
        # print(dict(single ))
        # print(single)
        ort = {
                'attention_mask': np.array(single['attention_mask'], dtype=np.int64), 
                'input_ids': np.array(single['input_ids'], dtype=np.int64), 
                'token_type_ids': np.array(single['token_type_ids'], dtype=np.int64)
        }
        # print(ort)
        outputs = session.run(None, ort )
        print(outputs[0] )
        if i==2:
            break
        
    
    # outputs = session.run(None, dict(dataset[0]))
    # print(outputs[0] )
    
    # dataset = load_dataset('csv',  data_files = string)
    # dataset_new = dataset['test']
    # dataset_new.set_format("pandas")
    # DF = dataset_new[:2]
    # _ = DF.pop('labels')
    # print(DF)
    # dictionary = {}
    # input_ids = DF['input_ids']
    # input_ids = np.array(input_ids)
    # attention_mask = DF['attention_mask']
    # attention_mask = np.array(attention_mask)
    # token_type_ids = DF['token_type_ids']
    # token_type_ids = np.array(token_type_ids)
    # dictionary['input_ids']= input_ids
    # dictionary['attention_mask']= attention_mask
    # dictionary['token_type_ids']= token_type_ids
    # inp = DF['input_ids']
    # dataset_No_Label = Dataset.from_pandas(DF)
    
    # print(dataset_No_Label)
    # Orginal_Model = onnx.load(output)
    # Orginal_Model=OnnxModel(Orginal_Model)
    # Compressed_Model = onnx.load(Compressed_Onnx_output_path)
    # Compressed_Model=OnnxModel(Compressed_Model)
    # Quantized_Model = onnx.load(Quantized_Onnx_output_path)
    # Quantized_Model=OnnxModel(Quantized_Model)
    # Compressed_opt_Model = onnx.load(Compressed_opt_Onnx_output_path)
    # Compressed_opt_Model=OnnxModel(Compressed_opt_Model)
    
    
    
    # # session = InferenceSession(Orginal_Model, options)
    # count = len(dataset_No_Label['input_ids']) 
    # for i in range(count):
    #     data = dataset_No_Label[i]
    #     # ort_inputs = {
    #     #                 'attention_mask':  data[0].cpu().numpy(),
    #     #                 'input_ids': data[1].cpu().numpy(),
    #     #                 'token_type_ids': data[2].cpu().numpy()
    #     #             }
    
    # dataset_new [0:2]
    
    
    
    # """Weights & Baises Enviorment Variable Initialization"""
    # """Used this Notebook for References: https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=XMQuuun2R06J"""
    # TrainingArgument = params['TrainingArgument']
    # WANDB_PROJECT = TrainingArgument['WANDB_PROJECT']
    # os.environ['WANDB_PROJECT'] = WANDB_PROJECT
    
    # """Onnx Model Convertion Starts"""
    # inputs, outputs, model = parameters(features)
    # mmodel = model(Best_path)
    # mmodel = mmodel.to('cpu')
    # # Generate dummy inputs
    # dummy = dict(tokenizer(["find flights arriving new york city next saturday"], return_tensors="pt").to('cpu'))
    # with torch.no_grad():
    #     mmodel.config.return_dict = True
    #     mmodel.eval()
    #     create_directory([Onnx_output_path]) 
    #     output = os.path.join(Onnx_output_path,"model.onnx")
    #     export(
    #             mmodel,
    #             (dummy,),
    #             output,
    #             opset_version=opset,
    #             do_constant_folding=True,
    #             input_names=list(inputs.keys()),
    #             output_names=list(outputs.keys()),
    #             dynamic_axes=dict(chain(inputs.items(), outputs.items())),
    #         )
     
    # logging.info(f"Onnx Model Created {output}")
    # Compressed_Onnx_output_path = os.path.join(Onnx_output_path,"model_compressed.onnx")
    # CompressModel(output,Compressed_Onnx_output_path)
    # Quantized_Onnx_output_path = os.path.join(Onnx_output_path,"model_Quantized.onnx")
    # quantize_onnx_model(Compressed_Onnx_output_path,Quantized_Onnx_output_path)
    
    # Orginal_Model = onnx.load(output)
    # Compressed_Model = onnx.load(Compressed_Onnx_output_path)
    # Quantized_Model = onnx.load(Quantized_Onnx_output_path)
    
    
    
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