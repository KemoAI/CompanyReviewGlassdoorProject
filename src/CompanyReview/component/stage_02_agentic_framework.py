import os
import re
import pandas as pd
import re
import asyncio
import aiohttp
from pathlib import Path
from tqdm import tqdm
import time
from CompanyReview import logger
from CompanyReview.entity.config_entity import AgenticFrameworkConfig
from CompanyReview.utils import utils

class GenerateReview:
    def __init__(self, config:AgenticFrameworkConfig):
        self.config = config
    
    def __get_num(self, text):
        """returns a string of the number"""
        try:
            return re.findall(r'\b[-+]?\d*\.*\d+\b', text)[0]        # return the matched number if possible
        except:
            return None                                           # if an error occurred
        
    def __is_int(self, str_num):
        """Checks if the string input represents and integer"""
        try:
            s = int(str_num)
            return True      # success -> return True
        except:              
            return False     # error -> return False
            
    def __extract_feedback(self , response):
        """the method extracts the feedback from the critic response"""
        Y_N_reply = response.split('\n')[0]
        remaining = "\n".join(response.split('\n')[1:]).strip()
        score     = self.__get_num(remaining.split('\n')[0])
        if score is not None and self.__is_int(score):
            score = int(score)/100
        elif score is not None:
            score = float(score)
        feedback  = "\n".join(remaining.split('\n')[1:]).strip()
        return Y_N_reply, score, feedback

    def __get_chat(self , composer_responses, critic_responses):
        """It formats the chat in a presentable manner"""
        s = ""
        for i, (composer_response, critic_response) in enumerate(zip(composer_responses, critic_responses)):
            s = s + f"\nComposer Answer {i+1}:\n"
            s = s + composer_response[0] + "\n\n"
            s = s + f"Critic Response: {critic_response[1]}\n"
            s = s + f"Score: {critic_response[2]}\n"
            s = s + f"Feedback {i+1}:\n"
            s = s + critic_response[0] + "\n"
        return s.strip()

    def __get_agent_stats(self , responses):
        """the method calculates the agent stats"""
        stats = {
                'input-tokens' : 0,
                'output-tokens': 0,
                'total-tokens' : 0,
                'time'         : 0.0
                }
        for response in responses:
            usage, response_time    = response[-2], response[-1]
            stats['input-tokens']  += usage['prompt_tokens']
            stats['output-tokens'] += usage['completion_tokens']
            stats['total-tokens']  += usage['total_tokens']
            stats['time']          += response_time
        return stats
            
    def __get_composer_agent(self):
        return {
                'base-url'   : self.config.composer_agent_url,
                'model-name' : self.config.composer_agent_model,
                'api-key'    : self.config.deepseek_api_key
            }

    def __get_critic_agent(self):
        return {
                'base-url'   : self.config.critic_agent_url,
                'model-name' : self.config.critic_agent_model,
                'api-key'    : self.config.openai_api_key
            }

    async def __call_llm_async(self , messages , model_config , temperature = 0.5):
        """Async version of call_llm function"""
        try:
            # Extract API type and model from the configuration dictionary
            base_url = model_config["base-url"]
            model    = model_config["model-name"]
            api_key  = model_config["api-key"]
            
            # Create an asynchronous HTTP session
            async with aiohttp.ClientSession() as session:
                # Set up the request headers for authorization and content type
                headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type" : "application/json"
                        }

                # Prepare the payload for the API request
                payload = {
                            "model"       : model,              # Specify the model to use
                            "messages"    : messages,           # User's query
                            "temperature" : temperature,        # Sampling temperature
                            }

                # Send a POST request to the chat completions endpoint
                async with session.post(f"{base_url}/chat/completions", json=payload, headers=headers) as response:
                    result = await response.json()  # Parse the JSON response

                    # Extract token usage information from the response
                    usage = {
                            'prompt_tokens'     : result['usage']['prompt_tokens'],           # Tokens used in the prompt
                            'completion_tokens' : result['usage']['completion_tokens'],       # Tokens used in the completion
                            'total_tokens'      : result['usage']['total_tokens']             # Total tokens used
                            }

                    # Return the generated content and usage statistics
                    return result['choices'][0]['message']['content'], usage

        except Exception as e:  # Handle exceptions and print error message
            print(f"Error in call_llm_async function for model {model} using {base_url} API: {e}")
            return "", {}  # Return empty response on error
        

    async def __async_generate_review_agentic(self , record, composer_configs, critic_configs, temperature):
        """Start conversation between composer and critic to generate the review"""
        critic_responses = []
        composer_responses = []
        scores = []
        i = 1
        try:
            start_time    = time.time()
            # start the initial message
            review, usage = await self.__call_llm_async([{"role": "user", "content": self.config.composer_agent_query.format(record['pros'], record['cons'])}],
                                                            model_config = composer_configs, temperature = self.config.temperature)
            review = review.strip()
            composer_responses.append((review, usage, time.time() - start_time))             # save the composer response

            critic_start_time = time.time()
            response, usage = await self.__call_llm_async([{"role": "user", "content": self.config.critic_agent_query.format(record['pros'], record['cons'], review)}],
                                                model_config = critic_configs, temperature = self.config.temperature)
            
            reply, score, feedback = self.__extract_feedback(response)  # extract the feedback along with the score
            critic_responses.append((feedback, reply, score, usage, time.time() - critic_start_time))   # store the response into the critic
            
            scores.append(score)
            # loop of max 10 trials enhancing the review 
            while reply[0].lower() == 'y' and i < 10:
                # build a history providing the original query with the pros & cons and constraints (no bullet points, etc) followed by the latest review;
                # then, mention the critic feedback regarding accuracy etc, allowing it to generate a new one
                messages = [{"role": "user", "content": self.config.composer_agent_query.format(record['pros'], record['cons'])},
                            {"role": "assistant", "content": review},
                            {"role": "user", "content": feedback}]
                composer_start_time = time.time()
                # generate a new review
                review, usage = await self.__call_llm_async(messages, model_config = composer_configs, temperature = temperature)
                review = review.strip()
                composer_responses.append((review, usage, time.time() - composer_start_time))
                
                critic_start_time = time.time()
                response, usage = await self.__call_llm_async([{"role": "user", "content": self.config.critic_agent_query.format(record['pros'], record['cons'], review)}],
                                                                model_config = critic_configs , temperature = temperature)
                reply, score, feedback = self.__extract_feedback(response)
                critic_responses.append((feedback, reply, score, usage, time.time() - critic_start_time))

                scores.append(score)
                i += 1                       # update the counter to limit the number of trials
            time_taken    = time.time() - start_time
            return critic_responses, composer_responses, scores, time_taken

        except Exception as e:
            # Print an error message if an exception occurs during the process
            print(f"Error in async_generate_review_agentic: {e}")

            # Return None for the rating, the time taken so far, and an empty dictionary for usage
            return [], [], [], time.time() - start_time
        
    async def __agentic_review_generation_task_generator(self , records, results, progress_bar, composer_configs, critic_configs):
        """a function that generates concurrent asynchronous calls to do review generation"""

        # Create a semaphore to limit the number of concurrent tasks
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        async def process_record(record):
            # Acquire a semaphore to limit concurrent processing
            async with semaphore:
                # Generate a review asynchronously for the given record
                critic_responses, composer_responses, scores, time_taken = \
                                await self.__async_generate_review_agentic(record , 
                                                                           composer_configs = composer_configs,
                                                                           critic_configs   = critic_configs,
                                                                           temperature      = self.config.temperature)
                
                # If we have a review, put it into the results and update the progress bar
                if len(composer_responses) > 0 :
                    record['company-review']         = composer_responses[-1][0]
                    record['full-chat']              = self.__get_chat(composer_responses, critic_responses)
                    record['total-time']             = time_taken
                    record['num-of-trials']          = len(critic_responses)
                    record['scores']                 = str(scores)                          # use ast.literal_eval(scores) to convert it back to python
                    composer_stats                   = self.__get_agent_stats(composer_responses)
                    critic_stats                     = self.__get_agent_stats(critic_responses)
                    record['composer-input-tokens']  = composer_stats['input-tokens']
                    record['composer-output-tokens'] = composer_stats['output-tokens']
                    record['composer-total-tokens']  = composer_stats['total-tokens']
                    record['composer-total-time']    = composer_stats['time']
                    record['critic-input-tokens']    = critic_stats['input-tokens']
                    record['critic-output-tokens']   = critic_stats['output-tokens']
                    record['critic-total-tokens']    = critic_stats['total-tokens']
                    record['critic-total-time']      = critic_stats['time']
                    record['total-input-tokens']     = record['composer-input-tokens']  + record['critic-input-tokens']
                    record['total-output-tokens']    = record['composer-output-tokens'] + record['critic-output-tokens']
                    record['total-tokens']           = record['composer-total-tokens']  + record['critic-total-tokens'] 
                    results.append(record)              # append the result
                    progress_bar.update(1)

        # Create a list of tasks for processing each record
        tasks = [asyncio.create_task(process_record(record)) for record in records]

        await asyncio.gather(*tasks)        # Wait for all tasks to complete

    async def generate_reviews(self , records_list):
        # Initialize results list
        results = []
        # Set up progress bars for generator
        progress_bar = tqdm(total=len(records_list), desc="Review Generation", unit="record", position=0, leave=True)

        # Start the task generator to process the records
        await self.__agentic_review_generation_task_generator(records_list, results , progress_bar, 
                                                              composer_configs = self.__get_composer_agent(),
                                                              critic_configs   = self.__get_critic_agent())
        # Close the progress bars
        progress_bar.close()
        return results

    def save_generated_reviews(self, pdf:pd.DataFrame):
        pdf.to_csv(self.config.generated_review_path , index=False)