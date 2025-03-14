from pathlib import Path

CONFIG_FILE_PATH=Path("configs/config.yaml")
PARAMS_FILE_PATH=Path("params.yaml")

COMPOSER_AGENT_QUERY = """You have been assigned the responsibility of reviewing companies. Your assignment is to craft a professional expressive and persuasive review of the company based on employees feedback. 'Pros' and 'Cons' make up the feedback. In this context, Pros' denotes the favorable aspects of the feedback, while 'Cons' denotes the unfavorable aspects.
Pros: {}
Cons: {}
Task: Write a company review based on the inputs listed above. Focus solely on the "Pros" and "Cons" when writing the review. The review should be one paragraph of a maximum of 150 words (free of bullet points) and (free of meta-information). Do not include any introductions, disclaimers, word count, self-referential notes, headings or references to the terms "Pros" and "Cons" in the review. Begin writing the review content directly. End the review content naturally without trailing text and avoid abrupt cutoffs or incomplete sentences. Do not enclose the review in quotation marks or any other unnecessary formatting.

Response: """

COMPOSER_AGENT_MODEL = "deepseek-reasoner"
COMPOSER_AGENT_URL   = "https://api.deepseek.com"
DEEPSEEK_KEY         = "Enter Your Deepseek API Key Here"

CRITIC_AGENT_QUERY   = """Pros: {}
Cons: {}
Review: {}

Evaluate the provided company "Review" to confirm whether it accurately reflects the sentiment expressed in the "Pros" and "Cons." Your evaluation should focus on the following criteria: Accuracy (Does it accurately represent sentiment, tone, and key points from "Pros" and "Cons"?), Inclusiveness (Does it cover significant aspect without overemphasizing nor omitting any points?), Clarity and Precision (Is it straightforward, concise, and free from vagueness?). Respond with Y if it needs improvement or N if it meets the criteria. In a newline, assign a percentage score (0-100) of how much the review aligns with the criteria. If your response is Y, provide specific feedback, suggestions, and concerns after another new line, focusing on how the review could be improved to better align with the criteria. Do not rewrite or attempt to modify the review in your response."""

CRITIC_AGENT_URL   = "https://api.openai.com/v1"
CRITIC_AGENT_MODEL = "gpt-4o"
OPENAI_KEY         = "Enter Your OpenAI API Key Here"


HUGGINGFACE_TOKEN  = "Enter Your Huggingface API Token Here"             
WANDB_TOKEN        = "Enter Your Wandb API Token Here"

















