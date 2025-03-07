import pandas as pd
import sys
import asyncio
from CompanyReview import logger
from CompanyReview.config import ConfigurationManager
from CompanyReview.component.stage_02_agentic_framework import GenerateReview



if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Start processing
logger.info(f"Dual Agent Framework started")

main_config       = ConfigurationManager()
agent_config      = main_config.get_agents_config()
agent_build       = GenerateReview(agent_config)
df                 = pd.read_csv(agent_config.processed_data_path)
df['index']       = df.index 
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
generated_reviews = loop.run_until_complete(agent_build.generate_reviews(df.to_dict("records")[:1]))
generated_reviews = pd.DataFrame(generated_reviews)
generated_reviews = generated_reviews.sort_values(by=['index'], ascending=True).reset_index(drop=True)
agent_build.save_generated_reviews(generated_reviews)

logger.info(f"Dual Agent Framework just ended")
