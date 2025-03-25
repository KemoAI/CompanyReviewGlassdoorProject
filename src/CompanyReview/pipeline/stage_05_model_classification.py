import time

from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_05_model_classification import ModelClassification   


logger.info(f"Classification started")

main_config                = ConfigurationManager()
classification_config      = main_config.get_classification_config()
classification_model       = ModelClassification(classification_config)
deberta_model , tokenizer  = classification_model.load_deberta()
dataset, full_dataset      = classification_model.load_llama_8B_dataset(do_clean=True , do_regression=False)
classification_model.train(deberta_model , tokenizer , dataset)

logger.info(f"Classification just finished")