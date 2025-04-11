from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_06_model_classification_inference import ModelClassificationInference

logger.info(f"Classification Inference started")

main_config                          = ConfigurationManager()
classification_inference_config      = main_config.get_classification_inference_config()
classification_inference_obj         = ModelClassificationInference(classification_inference_config)
classification_inference_pipline     = classification_inference_obj.get_pipeline()
classification_inference_dataset     = classification_inference_obj.get_dataset()
classification_inference_prediction  = classification_inference_obj.get_predictions(classification_inference_pipline , classification_inference_dataset[:10])
classification_inference_report      = classification_inference_obj.get_report(classification_inference_prediction['overall-ratings'] , classification_inference_prediction['prediction'])
classification_inference_obj.display_report(classification_inference_report)

logger.info(f"Classification just finished")
