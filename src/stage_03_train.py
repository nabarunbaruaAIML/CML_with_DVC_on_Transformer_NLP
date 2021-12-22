import argparse
import os
import shutil
from tqdm import tqdm
import logging
import json
import numpy as np
from datasets import load_from_disk,load_metric
from src.utils.all_utils import read_yaml
from transformers import AutoTokenizer,DataCollatorWithPadding, EarlyStoppingCallback
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

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
STAGE = 'Stage 04'
def main(config_path):
    
    config = read_yaml(config_path.config)
    params = read_yaml(config_path.params)
    model = params['model']
    model_name = model['base_model']
    use_fast =  model['use_fast']
    padding =  model['padding']
    max_length =  model['max_length']
    truncation =  model['truncation']
    
    artifacts = config['artifacts']
    artifacts_dir = artifacts['ARTIFACTS_DIR']
    best = artifacts['Best_Dir']
    Best_path = os.path.join(artifacts_dir,best) 
    best_state = Best_path + '/trainer_state.json'
    DownloadData =  artifacts['DOWNLOAD_DATA_DIR']
    DownloadData_path = os.path.join(artifacts_dir,DownloadData)   
    Dataset_dir = artifacts['Dataset_dir']
    Dataset_path = os.path.join(DownloadData_path ,Dataset_dir)
    base_model_dir = artifacts['BASE_MODEL_DIR']
    base_model_path = os.path.join(artifacts_dir,base_model_dir )
    
    LABEL_NUM_filename =  artifacts['LABEL_NUM']  
    LABEL_NUM_filename_path = os.path.join(DownloadData_path ,LABEL_NUM_filename)
    
    Jfile = open(LABEL_NUM_filename_path)
    Jdata = json.load(Jfile)
    Jfile.close()
    
    dataset = load_from_disk(Dataset_path) #('artifacts/Data/Dataset')
    # dataset = load_from_disk('artifacts/Data/Dataset/T1')
    # dataset.set_format(type='torch')
    logging.info(f"Loaded Dataset from path {Dataset_path} Succefully and Dataset = {dataset}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name ,cache_dir =  base_model_path, use_fast=use_fast)
    logging.info(f"Loaded Tokenizer of Model {model_name} Succefully !")
    
    Transformer_Model = AutoModelForSequenceClassification.from_pretrained(model_name ,cache_dir =  base_model_path, num_labels=Jdata['Number_of_Label'])
    logging.info(f"Loaded Model {model_name} Succefully !")
    
    data_collator = DataCollatorWithPadding(tokenizer,padding = padding, max_length= max_length)
    
    TrainingArgument = params['TrainingArgument']
    metric_name = TrainingArgument['metric_name']
    Output_Dir = TrainingArgument['Output_Dir']
    Evaluation_Strategy = TrainingArgument['Evaluation_Strategy']
    Save_Strategy = TrainingArgument['Save_Strategy']
    Learning_Rate = TrainingArgument['Learning_Rate']
    Per_Device_Train_Batch_Size = TrainingArgument['Per_Device_Train_Batch_Size']
    Per_Device_Eval_Batch_Size = TrainingArgument['Per_Device_Eval_Batch_Size']
    Save_Total_Limit = TrainingArgument['Save_Total_Limit']
    Num_Train_Epochs = TrainingArgument['Num_Train_Epochs']
    Weight_Decay = TrainingArgument['Weight_Decay']
    Load_Best_Model_At_End = TrainingArgument['Load_Best_Model_At_End']
    Warmup_Ratio = TrainingArgument['Warmup_Ratio']
    Early_Stopping_patience = TrainingArgument['Early_Stopping_patience']
    Cuda = TrainingArgument['Cuda']
    
    metric_loaded = load_metric(metric_name)
    
    logging.info(f"Loaded Metric {metric_loaded} Succefully !")
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
    
        return metric_loaded.compute(predictions=predictions, references=labels)
    
    args = TrainingArguments(
                                output_dir= Output_Dir,
                                evaluation_strategy = Evaluation_Strategy,
                                save_strategy = Save_Strategy,
                                learning_rate= Learning_Rate,
                                per_device_train_batch_size= Per_Device_Train_Batch_Size,
                                per_device_eval_batch_size= Per_Device_Eval_Batch_Size,
                                save_total_limit= Save_Total_Limit,
                                num_train_epochs= Num_Train_Epochs,
                                weight_decay= Weight_Decay,
                                load_best_model_at_end= Load_Best_Model_At_End,
                                warmup_ratio = Warmup_Ratio,
                                metric_for_best_model=metric_name,
                                no_cuda = Cuda
                            ) 
    
    trainer = Trainer(
                        Transformer_Model,
                        args,
                        train_dataset=dataset['train'],#train_dataset, #encoded_dataset["train"],
                        eval_dataset=dataset['test'],#val_dataset, #encoded_dataset[validation_key],
                        data_collator=data_collator,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics
                    )
    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience= Early_Stopping_patience))
    # global_step, train_loss, out_metrics = trainer.train()
    global_step, train_loss, out_metrics= trainer.train()
    
    trainer.save_model(Best_path)
    trainer.state.save_to_json(best_state)
    
    logging.info(f"Training Completed with Check Point {global_step} training Loass {train_loss} and Details{out_metrics}")
    evalu = trainer.evaluate()
    
    logging.info(f"Evalution Completed with Evalution Data {evalu}")
    
    # ttt = AutoModelForSequenceClassification.from_pretrained('artifacts/Best_Model')
    
    # print(ttt)
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