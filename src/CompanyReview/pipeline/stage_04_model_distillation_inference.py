import time
from accelerate import PartialState

from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_04_model_distillation_inference import ModelDistillationInference

logger.info(f"Inference started")

main_config                    = ConfigurationManager()
distillation_inference_model   = ModelDistillationInference(main_config)
distillation_inferred_dataset  = distillation_inference_model.get_dataset()
distributed_state              = PartialState()
num_processes                  = distributed_state.num_processes
distillation_inferred_pipline  = distillation_inference_model.get_pipeline(distributed_state.device)
distillation_inferred_batches  = distillation_inference_model.get_index_batches(distillation_inferred_dataset)
start_time                     = time.time()
distillation_inference_model.save_inferrance_result(distributed_state , distillation_inferred_pipline  , distillation_inferred_batches)

if distributed_state.is_last_process:
    distillation_inference_model.merge_files(start_time , num_processes )

logger.info(f"Inference just finished")
