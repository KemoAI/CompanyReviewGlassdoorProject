import pandas as pd
import sys
import time
from accelerate import PartialState

from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_03_model_distillation import ModelDistillation


# Start processing

logger.info(f"Finetuning started")

main_config       = ConfigurationManager()
finetune_config   = main_config.get_finetune_config()
finetune_model    = ModelDistillation(finetune_config)
model , tokenizer = finetune_model.get_model()
reviews_dataset   = finetune_model.get_dataset()
finetune_trainer  = finetune_model.initialize_trainer(model, reviews_dataset)
start_time        = time.time()
finetune_trainer.train()
total_time        = time.time() - start_time
finetune_model.save_finetune_logs(total_time, finetune_trainer)

logger.info(f"Finetuning ended")