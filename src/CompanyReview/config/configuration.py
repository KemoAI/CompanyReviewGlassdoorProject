from CompanyReview.utils.utils import read_yaml
from CompanyReview.entity.config_entity import DataSourceConfig , AgenticFrameworkConfig
from CompanyReview.constants import CONFIG_FILE_PATH , PARAMS_FILE_PATH , COMPOSER_AGENT_QUERY , COMPOSER_AGENT_MODEL , COMPOSER_AGENT_URL, CRITIC_AGENT_QUERY , CRITIC_AGENT_MODEL , CRITIC_AGENT_URL
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
                                                            generated_review_path    = gen_config.generated_review_path,
                                                            composer_agent_query     = COMPOSER_AGENT_QUERY,
                                                            composer_agent_model     = COMPOSER_AGENT_MODEL,
                                                            composer_agent_url       = COMPOSER_AGENT_URL,
                                                            deepseek_api_key         = "a",
                                                            critic_agent_query       = CRITIC_AGENT_QUERY,
                                                            critic_agent_model       = CRITIC_AGENT_MODEL,
                                                            critic_agent_url         = CRITIC_AGENT_URL,
                                                            openai_api_key           = "a",
                                                            max_concurrent_requests  = self.params.max_concurrent_requests,
                                                            temperature              = self.params.temperature
                                                         )
        return agentic_framework_config 