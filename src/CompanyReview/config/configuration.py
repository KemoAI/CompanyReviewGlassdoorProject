from CompanyReview.utils.utils import read_yaml
from CompanyReview.entity.config_entity import DataLoadingConfig
from CompanyReview.constants import CONFIG_FILE_PATH
from pathlib import Path
import os

class ConfigurationManager:
    def __init__(self , config_file_path = CONFIG_FILE_PATH):
        self.config = read_yaml(config_file_path)

    def get_data_loading_config(self) -> DataLoadingConfig:
        config = self.config.data_source
        data_ingestion_config = DataLoadingConfig(
            root_data_path = config.root_data_path ,
            processed_data_path = config.processed_data_path)
        return data_ingestion_config