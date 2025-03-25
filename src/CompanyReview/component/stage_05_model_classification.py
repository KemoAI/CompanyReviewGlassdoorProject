import os
from os import listdir
from os.path import join
import json
import numpy as np
import pandas as pd

import re
from tqdm import tqdm

import time

from datasets import Dataset , DatasetDict
from transformers import Trainer, TrainerCallback, TrainingArguments, pipeline, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from evaluate import load

import wandb
from huggingface_hub import login

from CompanyReview import logger
from CompanyReview.entity.config_entity import ClassificationConfig


class ModelClassification:
    def __init__(self , config: ClassificationConfig):
        self.config = config

    def __load_model(self , MODEL_NAME):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if 'roberta' in MODEL_NAME:
            model = AutoModelForSequenceClassification.from_pretrained(
                                                                        MODEL_NAME                              ,
                                                                        num_labels  = 5                         ,  
                                                                        device_map  = "auto"                    ,
                                                                        id2label    = {i: i+1 for i in range(5)},
                                                                        label2id    = {i+1: i for i in range(5)},
                                                                        cache_dir   = self.config.cache_dir     ,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                                                                        MODEL_NAME                               ,
                                                                        num_labels  = 5                          ,
                                                                        id2label    = {i: i+1 for i in range(5)} ,
                                                                        label2id    = {i+1: i for i in range(5)} ,
                                                                        cache_dir   = self.config.cache_dir      ,
                                                                        )
        return model, tokenizer

    def load_deberta(self):
        DEBERTA_MODEL = "microsoft/deberta-v3-large"
        return self.__load_model(DEBERTA_MODEL)
    
    # dataset
    def __clean_text_for_classification(self , text):
        text = re.sub("’", "'", text).strip()                     # replace ’ with normal '
        text = re.sub(r"(.*\.)[^.]*$", r"\1", text)               # remove the last sentence
        unwanted_patterns = [
                                r"\(\d+ [Ww]ords\)",
                                r"\([Ww]ord [Cc]ount: \d+\)",
                                r"\(\d+ [cC]haracters\)",
                                r"#*\(? ?Task:.*$",
                                r"[Tt]ask$",
                                r"\(?Note.*$",
                                r"[rR]ead [Mm]ore.*$",
                                r"[^a-zA-Z.0-9]+$"
                            ]
        for pattern in unwanted_patterns:
            text = re.sub(pattern, "", text).strip()
        # remove quotes
        return re.sub(r'^[\"\'](.+)[\"\']$', r"\1", text).strip()
    
    def __load_dataset(self, MODEL_PATH, do_clean = False, do_regression = False):
        """The method reads the inference files of the available subsets: train, validation, and test"""
        # read dataframes
        PATH       = MODEL_PATH
        CSV_Files  = ['InferredTrain/inference_train_set.csv', 'InferredTest/inference_test_set.csv', 'InferredVal/inference_validation_set.csv']
        CSV_names  = ['train', 'test', 'validation']
        data_sets  = {}
        for i, f in enumerate(CSV_Files):        # if f in os.listdir(PATH):
            data_sets[CSV_names[i]] = pd.read_csv(os.path.join(PATH, f))

        # select only necessary columns
        for subset in data_sets:
            data_sets[subset] = data_sets[subset][['pros', 'cons', 'generated-review', 'overall-ratings']]
            # rename the columns for training
            data_sets[subset] = data_sets[subset].rename(columns={'generated-review': 'text', 'overall-ratings': 'label'})
            if do_clean:
                data_sets[subset]['text'] = data_sets[subset]['text'].apply(self.__clean_text_for_classification)
            # for regression loss, it expects it to be float
            if do_regression:
                data_sets[subset]['label'] = data_sets[subset]['label'].astype(float)
            # the Loss expects the labels to be 0 -> 4 for classification
            else:
                data_sets[subset]['label'] = data_sets[subset]['label']-1
        # read dataset
        dataset = DatasetDict({ k: Dataset.from_pandas(v[['text', 'label']]) for k,v in data_sets.items()})
        # Get the labels back to its original form
        for subset in data_sets:
            if not do_regression:
                # the Loss expects the labels to be 0 -> 4
                data_sets[subset]['label'] = data_sets[subset]['label']+1
        
        # read the full dataset
        full_dataset = DatasetDict({k: Dataset.from_pandas(v) for k,v in data_sets.items()})
        return dataset, full_dataset
    
    def load_llama_8B_dataset(self, do_clean = False, do_regression = False):
        return self.__load_dataset('llama-3.1-8B-finetuned/', do_clean = do_clean, do_regression = do_regression)

    def __preprocess(self, dataset, tokenizer):
        tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True) , batched=True)
        data_collator     = DataCollatorWithPadding(tokenizer=tokenizer)
        return tokenized_dataset, data_collator

# METRICS
    def __compute_metrics(self, eval_pred):
        accuracy_calculator  = load("accuracy")
        f1_calculator        = load("f1")
        precision_calculator = load("precision")
        recall_calculator    = load('recall')
        predictions, labels  = eval_pred
        predictions          = np.argmax(predictions, axis=1)
        acc                  = accuracy_calculator.compute(predictions=predictions, references=labels)['accuracy']
        f1                   = f1_calculator.compute(predictions=predictions, references=labels, average="macro")['f1']
        recall               = recall_calculator.compute(predictions=predictions, references=labels, average="macro")['recall']
        precision            = precision_calculator.compute(predictions=predictions, references=labels, average="macro")['precision']
        return {'Accuracy': acc, 'F1':f1, 'Precision': precision, 'Recall': recall}
    
    def __get_args(self):
        deberta_training_args = TrainingArguments(
                                                    run_name                    = self.config.run_name                    , 
                                                    output_dir                  = self.config.output_dir                  ,
                                                    num_train_epochs            = self.config.num_epochs                  ,
                                                    learning_rate               = self.config.lr                          ,
                                                    warmup_ratio                = self.config.warmup_ratio                ,
                                                    lr_scheduler_type           = self.config.lr_scheduler_type           ,
                                                    per_device_train_batch_size = self.config.per_device_train_batch_size ,
                                                    per_device_eval_batch_size  = self.config.per_device_eval_batch_size  ,
                                                    eval_strategy               = self.config.eval_strategy               ,
                                                    eval_steps                  = self.config.eval_steps                  ,
                                                    save_steps                  = self.config.save_steps                  ,
                                                    logging_steps               = self.config.logging_steps               ,
                                                    save_strategy               = self.config.save_strategy               ,
                                                    save_total_limit            = self.config.save_limit                  ,
                                                    load_best_model_at_end      = self.config.use_best                    ,
                                                    report_to                   = self.config.report_to
                                                )
        return deberta_training_args
    
    def train(self, model, tokenizer, dataset):
        if self.config.report_to == "wandb":
            os.environ["WANDB_PROJECT"]  = self.config.wandb_project
        tokenized_dataset, data_collator = self.__preprocess(dataset, tokenizer)
        training_args                    = self.__get_args()
        
        trainer  = Trainer(
                            model             = model,
                            args              = training_args,
                            train_dataset     = tokenized_dataset["train"],
                            eval_dataset      = tokenized_dataset["validation"],
                            processing_class  = tokenizer,
                            data_collator     = data_collator,
                            compute_metrics   = self.__compute_metrics,
                          )

        start_time = time.time()
        trainer.train()
        total_time = time.time() - start_time
        if self.config.report_to == "wandb":
            wandb.finish()

        # save time
        hh = int(total_time//60//60)
        mm = int((total_time-(hh*60*60))//60)
        ss = total_time-(hh*60*60)-(mm*60)
        logger.info(f"Time taken is {hh}:{mm}:{ss:.2f}")
        with open(os.path.join("./" + self.config.output_dir,"time.txt"), "w") as text_file:
            text_file.write(f"Time taken is {hh:02}:{mm:02}:{ss:.2f}")

        # create a directory to save the history
        os.makedirs(os.path.join("./" + self.config.output_dir, "history_logs/"), exist_ok=True)
        # Save the history log for plotting
        with open(os.path.join("./" + self.config.output_dir, "history_logs/log_history.txt"), 'w') as f:
            json.dump(trainer.state.log_history, f)
        logger.info(f"Log history has been saved to: {os.path.join('./' + self.config.output_dir, 'history_logs/log_history.txt')}")