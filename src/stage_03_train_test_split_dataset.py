import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.all_utils import read_yaml,create_directory,read_data,save_json
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import Dataset

logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
# create_directory([log_dir])
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
STAGE = 'Stage 03'
def main(config_path):
    config = read_yaml(config_path.config)
    params = read_yaml(config_path.params)
    
    model = params['Dataset']
    test_size = model['test_size']
    
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)
    DownloadData_filename =  artifacts['DOWNLOAD_DATA_NAME']
    Dataset_dir = artifacts['Dataset_dir']
    DownloadData_filename_path = os.path.join(DownloadData_path ,DownloadData_filename)
    
    
    Dataset_path = os.path.join(DownloadData_path ,Dataset_dir)
    dataset = load_dataset('csv', data_files= DownloadData_filename_path,  split='train[:100%]' )#'./artifacts/Data/Data.csv')
    dataset = dataset.train_test_split(test_size= test_size)
    # print(dataset)
     
    dataset.save_to_disk(Dataset_path)
    logging.info(f"Saved Dataset to path {Dataset_path} Succefully and Dataset = {dataset}")
    
    
    # secret = read_yaml(config_path.secret)
    # pass
    

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