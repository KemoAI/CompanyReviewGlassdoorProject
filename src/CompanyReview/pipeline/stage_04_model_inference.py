import time
from accelerate import PartialState

from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_04_model_inference import ModelInference

logger.info(f"Inference started")

main_config       = ConfigurationManager()
inference_config  = main_config.get_inference_config()
inference_model   = ModelInference(inference_config)
inferred_dataset  = inference_model.get_dataset()
distributed_state = PartialState()
num_processes     = distributed_state.num_processes
inferred_pipline  = inference_model.get_pipeline(distributed_state.device)
inferred_batches  = inference_model.get_index_batches()
start_time        = time.time()
inference_model.save_inferrance_result(distributed_state , inferred_pipline  , inferred_batches)

if distributed_state.is_last_process:
    inference_model.merge_files(start_time , num_processes )

logger.info(f"Inference just finished")
