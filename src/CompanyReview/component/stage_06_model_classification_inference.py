import os
import re
from os.path import join 
from os import listdir

import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import pipeline
from evaluate import load

from CompanyReview import logger
from CompanyReview.entity.config_entity import ClassificationInferenceConfig
from CompanyReview.utils import utils

class ModelClassificationInference:
    def __init__(self , config: ClassificationInferenceConfig):
        self.config = config

    def get_pipeline(self):
        PATH = self.config.checkpoint_path
        checkpoint = ""
        for f in os.listdir(PATH):
            if "checkpoint" in f:
                checkpoint = os.path.join(PATH, f)
                break
        classifier = pipeline("sentiment-analysis", model=checkpoint)
        return classifier

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
    
    def get_dataset(self):
        """This method loads the dataset from a csv file and splits it"""
        data         = pd.read_csv(self.config.classified_data_path)
        data['text'] = data['text'].apply(self.__clean_text_for_classification)
        return data
    
    def get_predictions(self, classifier, data_df):
        records = data_df.to_dict('records')
        new_records = []
        for r in tqdm(records):
            prediction = classifier(r['text'], truncation=True)[0]['label']
            r['prediction'] = prediction
            new_records.append(r)
        return pd.DataFrame(new_records)
    
    def get_report(self, label, predictions):
        label                = np.array(label)
        predictions          = np.array(predictions)
        accuracy_calculator  = load("accuracy")
        f1_calculator        = load("f1")
        precision_calculator = load("precision")
        recall_calculator    = load('recall')
        acc                  = accuracy_calculator.compute(predictions=predictions, references=label)['accuracy']
        f1_detailed          = f1_calculator.compute(predictions=predictions, references=label, average=None)['f1']
        f1                   = f1_calculator.compute(predictions=predictions, references=label, average="macro")['f1']
        precision_detailed   = precision_calculator.compute(predictions=predictions, references=label, average=None)['precision']
        precision            = precision_calculator.compute(predictions=predictions, references=label, average="macro")['precision']
        recall_detailed      = recall_calculator.compute(predictions=predictions, references=label, average=None)['recall']
        recall               = recall_calculator.compute(predictions=predictions, references=label, average="macro")['recall']
        return {'accuracy': acc, 'f1':f1, 'precision': precision, 'recall':recall , 'f1-detailed': f1_detailed, 'precision-detailed': precision_detailed, 'recall-detailed':recall_detailed}
    
    def display_report(self, report):
        logger.info('='*25)
        logger.info(f" Accuracy: {report['accuracy']*100:.2f}%")
        logger.info(f" F1-score: {report['f1']:.4f}")
        logger.info(f"Precision: {report['precision']:.4f}")
        logger.info(f"   Recall: {report['recall']:.4f}")
        logger.info('-'*25)
        logger.info(f'Detailed:')
        for i,v in enumerate(report['f1-detailed']):
            logger.info(f'Rating {i+1} F1-score: {v:.4f}')
        logger.info('-'*25)
        for i,v in enumerate(report['precision-detailed']):
            logger.info(f'Rating {i+1} Precision: {v:.4f}')
        logger.info('-'*25)
        for i,v in enumerate(report['recall-detailed']):
            logger.info(f'Rating {i+1} Recall: {v:.4f}')
        logger.info('='*25)
                
