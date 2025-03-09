from CompanyReview.config.configuration import ConfigurationManager
from CompanyReview.component.stage_01_data_preparation import DataPreparation
from CompanyReview import logger

logger.info(f"Start Data Preprocessing")

main_config         = ConfigurationManager()
data_source_config  = main_config.get_data_source_config()
get_source_data     = DataPreparation(data_source_config)
data_df             = get_source_data.build_dataframe()
data_df             = get_source_data.data_engineering(data_df)
get_source_data.save_processed_data(data_df)

logger.info(f"Data Preprocessing just Ended")
