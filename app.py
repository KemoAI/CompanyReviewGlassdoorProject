import streamlit as st
import yaml
import re
import torch
import os
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, pipeline

COMPOSER_AGENT_QUERY = """### Task: You have been assigned the responsibility of reviewing companies. Your assignment is to craft a professional expressive and persuasive review of the company based on employees feedback. 'Pros' and 'Cons' make up the feedback. In this context, Pros' denotes the favorable aspects of the feedback, while 'Cons' denotes the unfavorable aspects.
Pros: {}
Cons: {}
Task: Write a company review based on the inputs listed above. Focus solely on the "Pros" and "Cons" when writing the review. The review should be one paragraph of a maximum of 150 words (free of bullet points) and (free of meta-information). Do not include any introductions, disclaimers, word count, self-referential notes, headings or references to the terms "Pros" and "Cons" in the review. Begin writing the review content directly. End the review content naturally without trailing text and avoid abrupt cutoffs or incomplete sentences. Do not enclose the review in quotation marks or any other unnecessary formatting.

### Response: """

def load_config():
    with open("./app_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # search for the checkpoint
    if "checkpoint-" not in config['composer_checkpoint_path']:
        for f in os.listdir(config['composer_checkpoint_path']):
            if "checkpoint-" in f:
                config['composer_checkpoint_path'] = os.path.join(config['composer_checkpoint_path'], f)
                break

    # search for the classifier checkpoint
    if "checkpoint-" not in config['classifier_checkpoint_path']:
        for f in os.listdir(config['classifier_checkpoint_path']):
            if "checkpoint-" in f:
                config['classifier_checkpoint_path'] = os.path.join(config['classifier_checkpoint_path'], f)
                break
    return config

def clean_text_for_classification(text):
    text = re.sub("’", "'", text).strip()                     # replace ’ with normal '
    text = re.sub(r"(.*\.)[^.]*$", r"\1", text)               # remove the last sentence
    unwanted_patterns = [
        r"\(\d+ [Ww]ords\)",
        r"\([Ww]ord [Cc]ount: \d+\)",
        r"\(\d+ [cC]haracters\)",
        r"#*\(? ?Task:.*$",
        r"[Tt]ask$",
        r"\(?Note.*$",
        r"[rR]ead [Mm]ore.*$",
        r"[^a-zA-Z.0-9]+$"
    ]
    for pattern in unwanted_patterns:
        text = re.sub(pattern, "", text).strip()
    # remove quotes
    return re.sub(r'^[\"\'](.+)[\"\']$', r"\1", text).strip()

def load_reviews_pipeline(config):
    parms = {}
    parms['pretrained_model_name_or_path'] = config['composer_checkpoint_path']
    parms['torch_dtype']                   = torch.float16
    parms['device_map']                    = config['device']
    parms['cache_dir']                     = config['cache_dir']
    model     = AutoPeftModelForCausalLM.from_pretrained(**parms)
    tokenizer = AutoTokenizer.from_pretrained(config['composer_checkpoint_path'])
    if tokenizer.pad_token is None:              # configure tokenizer
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pipe = pipeline("text-generation", model = model, tokenizer = tokenizer)
    return pipe

def infer_review(pipeline, pros, cons, config):
    prompt = COMPOSER_AGENT_QUERY.format(pros,cons)
    output = pipeline(prompt                                                 ,    # do the inference getting the output
                      max_new_tokens       = config['max_new_tokens']        , 
                      do_sample            = True                            ,
                      top_p                = config['top_p']                 , 
                      top_k                = config['top_k']                 , 
                      temperature          = config['temperature']           ,
                      num_beams            = config['num_beams']             ,
                      num_return_sequences = config['num_return_sequences']  ,
                      eos_token_id         = pipeline.tokenizer.eos_token_id ,
                      pad_token_id         = pipeline.tokenizer.pad_token_id )
    return clean_text_for_classification(output[0]['generated_text'][len(prompt):].strip())

def load_classifier_pipeline(config):
    classifier = pipeline("sentiment-analysis", model= config['classifier_checkpoint_path'] , device='cpu')
    return classifier

def classifier_inference(review, classifier):
    return classifier(review, truncation=True)[0]['label']

# Page title
st.markdown("<h3 style='text-align: center; font-weight: bold;'>Rating Company Based on Employee Feedback</h4>",
    unsafe_allow_html=True)

# Input fields
pros = st.text_area("Enter Pros", height=100)
cons = st.text_area("Enter Cons", height=100)

# Load configuration
main_config    = load_config()

# Initialize session state
if "review_output" not in st.session_state:
    st.session_state.review_output = ""
if "rating_output" not in st.session_state:
    st.session_state.rating_output = ""

# Generate Review
generate_review = st.button("Generate Review" , disabled=not (pros.strip() and cons.strip()))
review_output = ""
if generate_review:
    review_pipline                 = load_reviews_pipeline(main_config)
    st.session_state.review_output = infer_review(review_pipline , pros , cons, main_config) 
    del review_pipline
st.text_area("Generated Review", value=st.session_state.review_output, height=150, disabled=True)

# Generate Rating
generate_rating = st.button("Generate Rating" , disabled=not (st.session_state.review_output.strip()))
rating_output = ""
if generate_rating:
    classifier_pipline             = load_classifier_pipeline(main_config)
    st.session_state.rating_output = "⭐" * classifier_inference(st.session_state.review_output , classifier_pipline)
    del classifier_pipline

st.text_input("Rating", value=st.session_state.rating_output, disabled=True)


