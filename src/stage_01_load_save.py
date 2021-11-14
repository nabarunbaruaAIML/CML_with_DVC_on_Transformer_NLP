import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.all_utils import read_yaml, create_directory,copy_file_csv


# logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
log_dir = "logs"
# create_directory([log_dir])
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )
STAGE = 'Stage 01'
def main(config_path):
    config = read_yaml(config_path.config)
    # params = read_yaml(config_path.params)
    # secret = read_yaml(config_path.secret)
    artifacts = config['artifacts']
    source_download_dirs= config["source_download_dirs"]
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    local_data_dirs = config["local_data_dirs"] 
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)# This can be replaced by local_data_dirs variable. Just keeping it for understanding perpose.
    create_directory([artifacts_dir,DownloadData_path])
    for source_download_dir, local_data_dir in tqdm(zip(source_download_dirs, local_data_dirs), total=1, desc= "list of folders", colour="red"):
        create_directory([local_data_dir])
        logging.info(f"Copying Files from directory {source_download_dir} to {local_data_dir} Started")
        copy_file_csv(source_download_dir, local_data_dir)
    logging.info(f"Copying Files Completed Successfully")
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    # args.add_argument("--secret","-s", default="configs/secrets.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info(f">>>>> {STAGE} started <<<<<")
        main(parsed_args)
        logging.info(f">>>>> {STAGE}  completed! <<<<<\n\n")
    except Exception as e:
        logging.exception(e)
        raise e