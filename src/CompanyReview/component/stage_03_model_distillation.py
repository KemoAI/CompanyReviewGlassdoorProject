import os

import pandas as pd
import numpy as np
import argparse
import wandb
from huggingface_hub import login
import json

import torch
from trl import SFTTrainer
from transformers import AutoModelForCausalLM , AutoTokenizer , TrainingArguments
from datasets import Dataset , DatasetDict
from accelerate import PartialState
from peft import LoraConfig , TaskType ,  get_peft_model
from sklearn.model_selection import train_test_split

from CompanyReview import logger
from CompanyReview.entity.config_entity import FineTuningConfig
from CompanyReview.utils import utils

class ModelDistillation:
    def __init__(self, config: FineTuningConfig):
        self.config = config
        ratios_sum        = self.config.train_ratio + self.config.val_ratio + self.config.test_ratio
        normalized_ratios = np.array([self.config.train_ratio , self.config.val_ratio, self.config.test_ratio]) / ratios_sum
        self.config.train_ratio , self.config.val_ratio , self.config.test_ratio = normalized_ratios

    def get_model(self):

        login(self.config.huggingface_token)                                        # login to huggingface
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path)          # load the tokenizer and the model
        if self.config.distributed_training:
            device_string = PartialState().process_index
            model         = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, device_map = {'':device_string} , torch_dtype = torch.float16 , cache_dir = self.config.cache_dir)
        else:
            model         = AutoModelForCausalLM.from_pretrained(self.config.model_name_or_path, device_map = "auto", torch_dtype = torch.float16, cache_dir = self.config.cache_dir)
        
        # Configure Low Rank Adapters.
        lora_config = LoraConfig(
                                r            = self.config.lora_rank,     # LoRa rank
                                lora_alpha   = self.config.lora_alpha,    # A scaling factor for the LoRA updates. It controls the contribution of the LoRA parameters to the overall model output.
                                lora_dropout = self.config.lora_dropout,  # The dropout rate applied to the LoRA parameters during training. This helps prevent overfitting by randomly dropping some of the LoRA parameters.
                                bias         = "none",             # Specifies how to handle biases in the target modules. In this case, it indicates that no bias will be added to the LoRA layers.
                                task_type    = TaskType.CAUSAL_LM  # The task type of the target model. In this case, it is a causal language modeling task.
                                )
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
       
        # configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
        return model, tokenizer

    def get_dataset(self):
        """This method loads the dataset from a csv file and splits it"""

        data         = pd.read_csv(self.config.processed_data_path)
        data['text'] = data.apply(lambda record: self.config.composer_agent_query.format(record['pros'],record['cons']), axis=1)
        data         = data.rename(columns={"company-review": "label"})
        y            = data['overall-ratings'].tolist()
        data         = data[['text', 'label']].astype(str)

        indx         = np.arange(len(data))                    # split the data using the indices
        indx_train, indx_test_val, y_train, y_test_val = train_test_split(indx, y, test_size = self.config.val_ratio + self.config.test_ratio , random_state = self.config.seed, stratify = y)
        test_ratio_wrt_val                 = self.config.test_ratio/(self.config.val_ratio + self.config.test_ratio)          # get the test ratio with respect to the both test and validation
        indx_val, indx_test, y_val, y_test = train_test_split(indx_test_val, y_test_val, test_size=test_ratio_wrt_val, random_state = self.config.seed, stratify = y_test_val)

        reviews_dataset = DatasetDict({
                                        'train'      : Dataset.from_pandas(data.iloc[indx_train].reset_index(drop=True)) ,
                                        'validation' : Dataset.from_pandas(data.iloc[indx_val].reset_index(drop=True))   ,
                                        'test'       : Dataset.from_pandas(data.iloc[indx_test].reset_index(drop=True))  ,
                             })
        
        return reviews_dataset

    def __formatting_prompts_func(self, example):
        """Formats the prompts for LLM training
        reference: https://huggingface.co/docs/trl/sft_trainer#format-your-input-prompts
        """
        output_texts = []
        for i in range(len(example['text'])):
            text = f"### Task: {example['text'][i]}\n### Response: {example['label'][i]}"
            output_texts.append(text)
        return output_texts
    
    def initialize_trainer(self , model , tokenizer, dataset, formatting_func = None):
        if self.config.distributed_training:                 # multiple GPUs
            training_args = TrainingArguments(
                                                output_dir                    = self.config.output_dir,                     # experiment directory
                                                num_train_epochs              = self.config.epochs,                         # number of training epochs
                                                gradient_accumulation_steps   = self.config.gradient_accumulation_steps,    # number of steps before performing a backward/update pass
                                                per_device_train_batch_size   = self.config.per_device_train_batch_size,    # batch size per device during training
                                                per_device_eval_batch_size    = self.config.per_device_eval_batch_size,     # batch size per device during evaluation          
                                                gradient_checkpointing        = self.config.gradient_checkpointing,         # use gradient checkpointing to save memory
                                                eval_strategy                 = self.config.eval_strategy,                  # evaluate per number of steps
                                                eval_steps                    = self.config.eval_steps,                     # number of steps per evaluation
                                                logging_steps                 = self.config.logging_steps,                  # log every 10 steps
                                                save_strategy                 = self.config.save_strategy,                  # save checkpoint every epoch
                                                save_total_limit              = self.config.save_limit,                     # limit the number of checkpoints saved
                                                save_steps                    = self.config.save_steps,                     # save after how many steps
                                                learning_rate                 = self.config.lr,                             # learning rate, based on QLoRA paper, Llama
                                                warmup_steps                  = self.config.warmup_steps,                   # warmup steps
                                                lr_scheduler_type             = self.config.lr_scheduler_type,              # use cosine learning rate scheduler
                                                max_grad_norm                 = self.config.max_grad_norm,                  # max_grad_norm for clipping
                                                load_best_model_at_end        = self.config.use_best, 
                                                push_to_hub                   = self.config.push_to_hub,                    # push model to hub
                                                report_to                     = self.config.report_to,                      # report metrics to weights and biases
                                                gradient_checkpointing_kwargs = {'use_reentrant':False}              # this is needed in case of multiple GPUs: https://huggingface.co/docs/trl/main/en/sft_trainer#multi-gpu-training
                                     )
        else:
            training_args = TrainingArguments(
                                                output_dir                   = self.config.output_dir,                      # experiment directory
                                                num_train_epochs             = self.config.epochs,                          # number of training epochs           
                                                gradient_accumulation_steps  = self.config.gradient_accumulation_steps,     # number of steps before performing a backward/update pass
                                                per_device_train_batch_size  = self.config.per_device_train_batch_size,     # batch size per device during training
                                                per_device_eval_batch_size   = self.config.per_device_eval_batch_size,      # batch size per device during evaluation
                                                gradient_checkpointing       = self.config.gradient_checkpointing,          # use gradient checkpointing to save memory
                                                eval_strategy                = self.config.eval_strategy,                   # evaluate per number of steps
                                                eval_steps                   = self.config.eval_steps,                      # number of steps per evaluation
                                                logging_steps                = self.config.logging_steps,                   # log every 10 steps
                                                save_strategy                = self.config.save_strategy,                   # save checkpoint every epoch
                                                save_total_limit             = self.config.save_limit,                      # limit the number of checkpoints saved
                                                save_steps                   = self.config.save_steps,                      # save after how many steps
                                                learning_rate                = self.config.lr,                              # learning rate, based on QLoRA paper, Llama
                                                warmup_steps                 = self.config.warmup_steps,                    # warmup steps
                                                lr_scheduler_type            = self.config.lr_scheduler_type,               # use cosine learning rate scheduler
                                                max_grad_norm                = self.config.max_grad_norm,                   # max_grad_norm for clipping
                                                load_best_model_at_end       = self.config.use_best,
                                                push_to_hub                  = self.config.push_to_hub,                     # push model to hub
                                                report_to                    = self.config.report_to,                       # report metrics to weights and biases
                                            )
        # Setup trainer with metrics
        if formatting_func is None:
            formatting_func = self.config.__formatting_prompts_func

        trainer = SFTTrainer(
                                model           = model                 ,
                                args            = training_args         ,
                                train_dataset   = dataset['train']      ,
                                eval_dataset    = dataset['validation'] ,
                                formatting_func = formatting_func       ,
                                tokenizer       = tokenizer             ,
                             )
        return trainer

    def save_finetune_logs(self , total_time , finetune_trainer ):
        device_string = PartialState().process_index
        
        os.makedirs(f"./{self.config.output_dir}/history_logs/", exist_ok=True)
        with open(f"./{self.config.output_dir}/history_logs/log_history_{device_string}.txt", 'w') as f:
            json.dump(finetune_trainer.state.log_history, f)
            print(f"Log history has been saved to: ./{self.config.output_dir}/history_logs/log_history_{device_string}.txt")
        
        hh = int(total_time//60//60)
        mm = int((total_time-(hh*60*60))//60)
        ss = total_time-(hh*60*60)-(mm*60)
        print(f"Time taken is {hh}:{mm}:{ss:.2f}")
        
        with open(f"./{self.config.output_dir}/time_{device_string}.txt", "w") as text_file:
            text_file.write(f"Time taken is {hh:02}:{mm:02}:{ss:.2f}")



