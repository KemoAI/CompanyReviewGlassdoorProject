from CompanyReview.utils.utils import read_yaml
from CompanyReview.entity.config_entity import DataSourceConfig , AgenticFrameworkConfig , FineTuningConfig , DistillationInferenceConfig, ClassificationConfig, ClassificationInferenceConfig
from CompanyReview.constants import CONFIG_FILE_PATH , PARAMS_FILE_PATH , COMPOSER_AGENT_QUERY , COMPOSER_AGENT_MODEL , COMPOSER_AGENT_URL, DEEPSEEK_KEY , CRITIC_AGENT_QUERY , CRITIC_AGENT_MODEL , CRITIC_AGENT_URL, OPENAI_KEY, HUGGINGFACE_TOKEN, WANDB_TOKEN
from pathlib import Path
import os

class ConfigurationManager:
    def __init__(self , config_file_path = CONFIG_FILE_PATH , params_file_path = PARAMS_FILE_PATH):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)

    def get_data_source_config(self) -> DataSourceConfig:
        config = self.config.data_source
        data_source_config = DataSourceConfig(
                                                  root_data_path      = config.root_data_path ,
                                                  processed_data_path = config.processed_data_path
                                             )
        return data_source_config
    
    def get_agenticframework_config(self) -> AgenticFrameworkConfig:
        src_config = self.config.data_source
        gen_config = self.config.generated_reviews
        agentic_framework_config = AgenticFrameworkConfig(
                                                            processed_data_path      = src_config.processed_data_path,
                                                            generated_review_path    = gen_config.generated_reviews_path,
                                                            composer_agent_query     = COMPOSER_AGENT_QUERY,
                                                            composer_agent_model     = COMPOSER_AGENT_MODEL,
                                                            composer_agent_url       = COMPOSER_AGENT_URL,
                                                            deepseek_api_key         = DEEPSEEK_KEY,
                                                            critic_agent_query       = CRITIC_AGENT_QUERY,
                                                            critic_agent_model       = CRITIC_AGENT_MODEL,
                                                            critic_agent_url         = CRITIC_AGENT_URL,
                                                            openai_api_key           = OPENAI_KEY,
                                                            max_concurrent_requests  = self.params.MAX_CONCURRENT_REQUESTS,
                                                            temperature              = self.params.TEMPERATURE,
                                                         )
        return agentic_framework_config
    
    def get_finetune_config(self):
        gen_config = self.config.generated_reviews
        finetune_config  = FineTuningConfig (
                                                model_name_or_path          = self.params.MODEL_NAME_OR_PATH           ,
                                                cache_dir                   = self.params.CACHE_DIR                    ,
                                                lora_rank                   = self.params.LORA_RANK                    ,                  # lora flags
                                                lora_alpha                  = self.params.LORA_ALPHA                   ,
                                                lora_dropout                = self.params.LORA_DROPOUT                 ,  
                                                generated_data_path         = gen_config.generated_reviews_path        ,                  # data flags
                                                composer_agent_query        = COMPOSER_AGENT_QUERY                     ,
                                                train_ratio                 = self.params.TRAIN_RATIO                  ,
                                                val_ratio                   = self.params.VAL_RATIO                    ,
                                                test_ratio                  = self.params.TEST_RATIO                   ,
                                                seed                        = self.params.SEED                         ,
                                                output_dir                  = self.params.OUTPUT_DIR                   ,                  # training args
                                                num_epochs                  = self.params.EPOCHS                       ,
                                                gradient_accumulation_steps = self.params.GRADIENT_ACCUMULATION_STEPS  ,
                                                per_device_train_batch_size = self.params.PER_DEVICE_TRAIN_BATCH_SIZE  ,
                                                per_device_eval_batch_size  = self.params.PER_DEVICE_EVAL_BATCH_SIZE   ,
                                                gradient_checkpointing      = self.params.GRADIENT_CHECKPOINTING       ,
                                                eval_strategy               = self.params.EVAL_STRATEGY                ,
                                                eval_steps                  = self.params.EVAL_STEPS                   ,
                                                logging_steps               = self.params.LOGGING_STEPS                ,
                                                save_strategy               = self.params.SAVE_STRATEGY                ,
                                                save_limit                  = self.params.SAVE_LIMIT                   ,
                                                save_steps                  = self.params.SAVE_STEPS                   ,
                                                lr                          = self.params.LR                           ,
                                                warmup_steps                = self.params.WARMUP_STEPS                 ,
                                                lr_scheduler_type           = self.params.LR_SCHEDULER_TYPE            ,
                                                max_grad_norm               = self.params.MAX_GRAD_NORM                ,
                                                use_best                    = self.params.USE_BEST                     ,
                                                push_to_hub                 = self.params.PUSH_TO_HUB                  ,
                                                report_to                   = self.params.REPORT_TO                    ,
                                                distributed_training        = self.params.DISTRIBUTED_TRAINING         ,
                                                huggingface_token           = HUGGINGFACE_TOKEN             ,
                                                wandb_token                 = WANDB_TOKEN
                                         ) 

        return finetune_config 
    
    def get_distillation_inference_config(self):
        gen_config = self.config.generated_reviews
        inference_distillation_config = DistillationInferenceConfig(
                                                                    checkpoint_path      = self.params.CHECKPOINT_PATH       ,
                                                                    cache_dir            = self.params.CACHE_DIR             ,
                                                                    baseline             = self.params.BASELINE              ,
                                                                    use_quantization     = self.params.USE_QUANTIZATION      ,
                                                                    generated_data_path  = gen_config.generated_reviews_path ,
                                                                    composer_agent_query = COMPOSER_AGENT_QUERY              ,
                                                                    train_ratio          = self.params.TRAIN_RATIO           ,
                                                                    val_ratio            = self.params.VAL_RATIO             ,
                                                                    test_ratio           = self.params.TEST_RATIO            ,
                                                                    seed                 = self.params.SEED                  ,
                                                                    inference_subset     = self.params.INFERENCE_SUBSET      ,
                                                                    results_path         = self.params.RESULTS_PATH          ,
                                                                    per_device_batch_size= self.params.PER_DEVICE_BATCH_SIZE ,
                                                                    max_new_tokens       = self.params.MAX_NEW_TOKENS        ,
                                                                    temperature          = self.params.TEMPERATURE           ,
                                                                    top_p                = self.params.TOP_P                 ,
                                                                    top_k                = self.params.TOP_K                 ,
                                                                    num_return_sequences = self.params.NUM_RETURN_SEQUENCES  ,
                                                                    num_beams            = self.params.NUM_BEAMS             ,
                                                                    repetition_penalty   = self.params.REPETITION_PENALTY    ,
                                                                    no_repeat_ngram_size = self.params.NO_REPEAT_NGRAM_SIZE  ,
                                                                    huggingface_token    = HUGGINGFACE_TOKEN
                                                                )
        return inference_distillation_config

    def get_classification_config(self):
        inferred_config       = self.config.inferred_reviews
        classification_config = ClassificationConfig(
                                                        wandb_project               = self.params.WANDB_PROJECT,
                                                        run_name                    = self.params.RUN_NAME,
                                                        inferred_path               = inferred_config.inferred_reviews_path,
                                                        output_dir                  = 'deberta-finetuned-from-llama-8b',
                                                        num_epochs                  = 4,
                                                        lr                          = 3e-6,
                                                        warmup_ratio                = self.params.WARMUP_RATIO,
                                                        lr_scheduler_type           = self.params.LR_SCHEDULER_TYPE,
                                                        per_device_train_batch_size = 16,
                                                        per_device_eval_batch_size  = 16,
                                                        eval_strategy               = self.params.EVAL_STRATEGY,
                                                        eval_steps                  = self.params.EVAL_STEPS,
                                                        save_steps                  = self.params.SAVE_STEPS, 
                                                        logging_steps               = self.params.LOGGING_STEPS, 
                                                        save_strategy               = self.params.SAVE_STRATEGY,
                                                        save_limit                  = self.params.SAVE_LIMIT,
                                                        use_best                    = self.params.USE_BEST,
                                                        report_to                   = self.params.REPORT_TO,
                                                        cache_dir                   = self.params.CACHE_DIR
                                                    )
        return classification_config
    
    def get_classification_inference_config(self):
        classification_inference_config = ClassificationInferenceConfig(
                                                                          checkpoint_path      = 'deberta-finetuned-from-llama-8b' ,
                                                                          classified_data_path = self.params.CLASSIFIED_DATA_PATH 
                                                                        )
        return classification_inference_config


    
