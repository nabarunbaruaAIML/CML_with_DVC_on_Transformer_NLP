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
>> 4 >> This is the Stage where we convert trained Weights to Onnx Format.
"""
STAGE = 'Stage 04'
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
    Best_path = os.path.join(artifacts_dir,best)       
    
    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_path = os.path.join(artifacts_dir,base_model_dir )
       
    """Loading Tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(Best_path ,cache_dir =  base_model_path, use_fast=use_fast)
    logging.info(f"Loaded Tokenizer of Model {model_name} Succefully !")
        
    """Weights & Baises Enviorment Variable Initialization"""
    """Used this Notebook for References: https://colab.research.google.com/github/neuml/txtai/blob/master/examples/18_Export_and_run_models_with_ONNX.ipynb#scrollTo=XMQuuun2R06J"""
    TrainingArgument = params['TrainingArgument']
    WANDB_PROJECT = TrainingArgument['WANDB_PROJECT']
    os.environ['WANDB_PROJECT'] = WANDB_PROJECT
    
    """Onnx Model Convertion Starts"""
    inputs, outputs, model = parameters(features)
    mmodel = model(Best_path)
    mmodel = mmodel.to('cpu')
    # Generate dummy inputs
    dummy = dict(tokenizer(["find flights arriving new york city next saturday"], return_tensors="pt").to('cpu'))
    with torch.no_grad():
        mmodel.config.return_dict = True
        mmodel.eval()
        create_directory([Onnx_output_path]) 
        output = os.path.join(Onnx_output_path,"model.onnx")
        export(
                mmodel,
                (dummy,),
                output,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(inputs.keys()),
                output_names=list(outputs.keys()),
                dynamic_axes=dict(chain(inputs.items(), outputs.items())),
            )
     
    logging.info(f"Onnx Model Created {output}")
    Compressed_Onnx_output_path = os.path.join(Onnx_output_path,"model_compressed.onnx")
    CompressModel(output,Compressed_Onnx_output_path)
    Quantized_Onnx_output_path = os.path.join(Onnx_output_path,"model_Quantized.onnx")
    quantize_onnx_model(Compressed_Onnx_output_path,Quantized_Onnx_output_path)
    
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