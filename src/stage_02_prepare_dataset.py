import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.all_utils import read_yaml,create_directory,read_dataset,save_json
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from datasets import Dataset

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
 
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
STAGE = 'Stage 02'
def main(config_path):
    config = read_yaml(config_path.config)
    local_data_dirs= config['local_data_dirs'][0]
    
    # Configuration Initialization
    artifacts = config['artifacts']
    local_data_dirs_filename = os.path.join(local_data_dirs ,artifacts['local_data_dirs_filename'])
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_path = os.path.join(artifacts_dir,base_model_dir )
    
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    Dataset_dir = artifacts['Dataset_dir']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)
    DownloadData_filename =  artifacts['DOWNLOAD_DATA_NAME']
    DownloadData_filename_path = os.path.join(DownloadData_path ,DownloadData_filename)
    Dataset_path = os.path.join(DownloadData_path ,Dataset_dir)
    ID2LABEL_filename =  artifacts['ID2LABEL']
    ID2LABEL_filename_path = os.path.join(DownloadData_path ,ID2LABEL_filename)
    
    LABEL2ID_filename =  artifacts['LABEL2ID']
    LABEL2ID_filename_path = os.path.join(DownloadData_path ,LABEL2ID_filename)
    
    LABEL_NUM_filename =  artifacts['LABEL_NUM']  
    LABEL_NUM_filename_path = os.path.join(DownloadData_path ,LABEL_NUM_filename)
    
    create_directory([base_model_path,DownloadData_path,Dataset_path])
    
    # Parameter Initialization
    params = read_yaml(config_path.params)
    Data = params['Dataset']
    test_size = Data['test_size']
    model = params['model']
    model_name = model['base_model']
    use_fast =  model['use_fast']
    padding =  model['padding']
    max_length =  model['max_length']
    truncation =  model['truncation']
    # secret = read_yaml(config_path.secret)
    
    # Prepareing and Saving Dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast,cache_dir = base_model_path )
    # json,id2label,label2id,label_num,Label_set = read_data(local_data_dirs_filename,tokenizer,padding,max_length,truncation)
    dataset,id2label,label2id,label_num,Label_set = read_dataset(local_data_dirs_filename,tokenizer,padding,max_length,truncation)
    label_num_json = {}
    label_num_json['Number_of_Label'] = label_num
    # save_json(DownloadData_filename_path,json)
    save_json(ID2LABEL_filename_path,id2label)
    save_json(LABEL2ID_filename_path,label2id)
    save_json(LABEL_NUM_filename_path,label_num_json)
    
    # dataset = Dataset.from_dict(json)
    dataset = dataset.train_test_split(test_size=test_size)
    dataset.save_to_disk(Dataset_path)#('artifacts/Data/Dataset/T1')
    logging.info(f"Saved transition Dataset into path {Dataset_path} Succefully!")
    # df = pd.DataFrame(json)
    # df.to_csv(DownloadData_filename_path,index= False)
    
    
    # dataset = load_dataset('csv', data_files='./artifacts/Data/Data.csv')
    # datasets.load_from_disk  DownloadData_path
    # dataset.save_to_disk  DownloadData_path
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.save_to_disk
    # https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets
    

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