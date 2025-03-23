from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataSourceConfig:
    root_data_path:Path
    processed_data_path:Path

@dataclass(frozen=True)
class AgenticFrameworkConfig:
    processed_data_path: Path
    generated_review_path: Path
    composer_agent_query: str
    composer_agent_model: str
    composer_agent_url: str
    deepseek_api_key: str
    critic_agent_query: str
    critic_agent_model: str
    critic_agent_url: str
    openai_api_key: str
    max_concurrent_requests: int
    temperature: float

@dataclass(frozen=False)
class FineTuningConfig:
    model_name_or_path: str
    cache_dir: str
    lora_rank: int                            #LoRa flags
    lora_alpha: int
    lora_dropout: float  
    processed_data_path: str                  #Data flags
    composer_agent_query: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    output_dir: str                    #Training args
    epochs: int
    gradient_accumulation_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_checkpointing: bool
    eval_strategy: str
    eval_steps: int
    logging_steps: int
    save_strategy: str
    save_limit: int
    save_steps: int
    lr: float
    warmup_steps: int
    lr_scheduler_type: str
    max_grad_norm: float
    use_best: bool
    push_to_hub: bool
    report_to: str
    distributed_training: bool
    huggingface_token: str             # Tokens
    wandb_token: str

@dataclass(frozen=False)
class InferenceConfig:
    checkpoint_path: str
    cache_dir: str
    baseline: bool
    use_quantization: bool
    generated_data_path: str                  #Data flags
    composer_agent_query: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    seed: int
    inference_subset: list
    results_path: str
    per_device_batch_size: int
    max_new_tokens: int 
    temperature: float
    top_p: float
    top_k: int
    num_return_sequences: int
    num_beams: int
    repetition_penalty: float
    no_repeat_ngram_size: int
    huggingface_token: str