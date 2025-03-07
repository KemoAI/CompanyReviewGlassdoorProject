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