from CompanyReview.config.configuration import ConfigurationManager
from CompanyReview.component.stage_01_data_preparation import DataPreparation
from CompanyReview import logger

logger.info(f"Start Data Preprocessing")

loading_config = ConfigurationManager()
loading_data_config = loading_config.get_data_loading_config()
loading_dataframe = DataPreparation(loading_data_config)
data_df = loading_dataframe.build_dataframe()
data_df = loading_dataframe.dataframe_preprocessing(data_df)
loading_dataframe.save_dataframe(data_df)

logger.info(f"Data Preprocessing just Ended")
