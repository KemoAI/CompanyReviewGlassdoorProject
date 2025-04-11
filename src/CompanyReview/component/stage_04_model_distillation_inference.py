import os
from os.path import join 
from os import listdir
import time

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from datasets import Dataset, DatasetDict
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from sklearn.model_selection import train_test_split

import wandb
from huggingface_hub import login

from CompanyReview import logger
from CompanyReview.entity.config_entity import DistillationInferenceConfig
from CompanyReview.utils import utils

class ModelDistillationInference:
    def __init__(self, config: DistillationInferenceConfig):
        self.config       = config
        ratios_sum        = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        normalized_ratios = np.array([self.config.train_ratio , self.config.val_ratio, self.config.test_ratio]) / ratios_sum
        self.config.train_ratio , self.config.val_ratio , self.config.test_ratio = normalized_ratios
        
        if not self.config.baseline and "checkpoint-" not in self.config.checkpoint_path:
            for f in os.listdir(self.config.checkpoint_path):
                if "checkpoint-" in f:
                    self.config.checkpoint_path = os.path.join(self.config.checkpoint_path, f)
                    logger.info(f'Loaded checkpoint: {self.config.checkpoint_path}')
                    break

    def __get_batches(self, items, batch_size):
        """The method splits the list of items into set of batches"""
        if batch_size == 1:
            return [[i] for i in items]                              # Get the batch size as the ceil of (number items / batch size)
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches     = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index   = min((i + 1) * batch_size, len(items))
            batch       = items[start_index:end_index]
            batches.append(batch)
        return batches
    
    def get_dataset(self):
        """This method loads the dataset from a csv file and splits it"""
        data         = pd.read_csv(self.config.generated_data_path)     # read the dataset
        data['text'] = data.apply( lambda record: self.config.composer_agent_query.format(record['pros'],record['cons']), axis=1)
        data         = data.rename(columns={"company-review": "label"})
        y            = data['overall-ratings'].tolist()
        data         = data[['pros', 'cons', 'label', 'text', 'overall-ratings']]
        indx         = np.arange(len(data))                                    # split the data using the indices
        indx_train, indx_test_val, y_train, y_test_val = train_test_split(indx, y, test_size = self.config.val_ratio + self.config.test_ratio, random_state = self.config.seed, stratify = y)
        test_ratio_wrt_val = self.config.test_ratio/(self.config.val_ratio + self.config.test_ratio)
        indx_val, indx_test, y_val, y_test = train_test_split(indx_test_val, y_test_val, test_size=test_ratio_wrt_val, random_state=self.config.seed, stratify=y_test_val)
       
        # split the dataset
        full_dataset = DatasetDict({
                                    'train'     : Dataset.from_pandas(data.iloc[indx_train].reset_index(drop=True)),
                                    'validation': Dataset.from_pandas(data.iloc[indx_val].reset_index(drop=True)),
                                    'test'      : Dataset.from_pandas(data.iloc[indx_test].reset_index(drop=True)),
                })
       
        # return only the requested subsets after turning them into list of dictionaries
        return [pd.DataFrame(full_dataset[subset]).to_dict('records') for subset in self.config.inference_subset]


    def get_pipeline(self, device):
        """The methods loads the model"""    
        parms = {}
        parms['pretrained_model_name_or_path'] = self.config.checkpoint_path
        parms['torch_dtype']                   = torch.float16
        parms['device_map']                    = device
        parms['cache_dir']                     = self.config.cache_dir
        if self.config.use_quantization:        # add quantization parameters if we are using quantized
            bnb_config = BitsAndBytesConfig(
                                            load_in_4bit              = True   ,
                                            bnb_4bit_use_double_quant = True   ,
                                            bnb_4bit_quant_type       = "nf4"  ,             # recommended by paper
                                            bnb_4bit_compute_dtype    = torch.bfloat16
                                           )
            parms['torch_dtype']         = torch.bfloat16
            parms['quantization_config'] = bnb_config

        if self.config.baseline:
            login(self.config.huggingface_token)                                 # if we are doing the inference for the baseline no LoRa
            model = AutoModelForCausalLM.from_pretrained(**parms)
        else:
            model = AutoPeftModelForCausalLM.from_pretrained(**parms)            # if we are doing the infere for finetuned model with LoRa parts
        tokenizer = AutoTokenizer.from_pretrained(self.config.checkpoint_path)
        if tokenizer.pad_token is None:              # configure tokenizer
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
        return pipe
    
    def get_index_batches(self, list_dataset):
        data = [(i,record) for i,record in enumerate(list_dataset[0])]
        batches = self.__get_batches(data, self.config.per_device_batch_size)
        return batches
    
    def save_inferrance_result(self, distributed_state , pipeline , batches):
        
        results_output = []
        with distributed_state.split_between_processes(batches ) as gpu_batch:
        
            for batch in tqdm(gpu_batch):                                                        # split_between_processes distributes the batches into the number of GPUs
                inputs = [r[1]['text'] for r in batch]
                
                outputs = pipeline(inputs                                                   ,    # do the inference getting the output
                                   max_new_tokens       = self.config.max_new_tokens        , 
                                   do_sample            = True                              ,
                                   top_p                = self.config.top_p                 , 
                                   top_k                = self.config.top_k                 , 
                                   temperature          = self.config.temperature           ,
                                   num_beams            = self.config.num_beams             ,
                                   num_return_sequences = self.config.num_return_sequences  ,
                                   eos_token_id         = pipeline.tokenizer.eos_token_id   ,
                                   pad_token_id         = pipeline.tokenizer.pad_token_id)
                results = [out[0]['generated_text'][len(inpt):].strip() for out,inpt in zip(outputs, inputs)]
                
                for res, (i,record) in zip(results, batch): # add the results
                    record['generated-review'] = res
                    results_output.append((i,record))
        
        results_output.sort(key=lambda x: x[0])             # sort the records again
      
        records = []                                        # add the index column    
        for i , r in results_output:
            r['indx'] = i
            records.append(r)
        
        s = distributed_state.process_index
        os.makedirs(self.config.results_path , exist_ok = True)
        pd.DataFrame(records).to_csv(join(self.config.results_path , f'inference_outputs_{s}.csv'), index = False)
    
    def merge_files(self, start_time, num_processes):
        
        csv_files = [f'inference_outputs_{i}.csv' for i in range(num_processes)]
        all_is_done = False

        while not all_is_done:                                                   # loop confirming if as long as not all the files are saved                   
            process_count = 0
            for f in csv_files:
                if f in listdir(self.config.results_path):
                    process_count += 1
            if process_count == num_processes:
                all_is_done = True
                break
            else:
                time.sleep(5)

        total_time = time.time() - start_time                                     # calculate the time needed so far

        dfs = [pd.read_csv(join(self.config.results_path, f)) for f in csv_files] # read each GPU intermediate file
        merged_records = pd.concat(dfs).to_dict('records')
        merged_records.sort(key = lambda x: int(x['indx']))                         # sort all the records across all the dataset
        
        df = pd.DataFrame(merged_records)
        df = df.drop('indx', axis=1)

        out_path = join(self.config.results_path, f'inference_{self.config.inference_subset[0]}_set.csv')
        df.to_csv(out_path, index=False)
        logger.info(f'Merged {len(csv_files)} files to {out_path}')

        hh = int(total_time//60//60)
        mm = int((total_time-(hh*60*60))//60)
        ss = total_time-(hh*60*60)-(mm*60)
        logger.info(f"Time taken is {hh}:{mm}:{ss:.2f}")

        with open(join(self.config.results_path, f'inference_{self.config.inference_subset[0]}_time.txt'), "w") as text_file:
            text_file.write(f"Time taken is {hh:02}:{mm:02}:{ss:.2f}")