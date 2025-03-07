import os
import re
import pandas as pd
from CompanyReview import logger
from pathlib import Path
from tqdm import tqdm
from CompanyReview.entity.config_entity import DataSourceConfig
from CompanyReview.utils import utils

class DataPreparation:
    def __init__(self, config:DataSourceConfig):
        self.config = config
    
    def build_dataframe(self):
        if not os.path.exists(self.config.root_data_path):
            logger.info("File is Missing")
            return None
        df = pd.read_csv(self.config.root_data_path)
        return df
    
    def __clean_text(self, text):
        # Handling Emojis https://unicode.scarfboy.com/
        # Grinning face with smiling eyes [-5]
        text = re.sub('\xf0\x9f\x98\x81', 'grinning_face_with_smiling_eyes_emoji ', text)
        # Grinning face with star eyes
        text = re.sub('\xf0\x9f\xa4\xa9', 'grinning_face_with_star_eyes_emoji ', text)
        # Smiling face with smiling eyes [-8]
        text = re.sub('\xf0\x9f\x98\x8a', 'smiling_face_with_smiling_eyes_emoji ', text)
        # OK sign emoji [-1]
        text = re.sub('\xf0\x9f\x91\x8c', 'OK_emoji ', text)
        # thumbs up emoji [-6]
        text = re.sub('\xf0\x9f\x91\x8d', 'thumbs_up_emoji ', text)
        # thumbs down
        text = re.sub('\xf0\x9f\x91\x8e', 'thumbs_down_emoji ', text)
        # robot face [-1]
        text = re.sub('\xf0\x9f\xa4\x96', 'robot_face_emoji ', text)
        # 100 points
        text = re.sub('\xf0\x9f\x92\xaf', '100_points_emoji ', text)
        # emoji modifier
        text = re.sub('\xf0\x9f\x8f\xbb', '', text)
        # emoji modifier fitzpatrick type-3
        text = re.sub('\xf0\x9f\x8f\xbc', '', text)
        # emoji modifier fitzpatrick type-4
        text = re.sub('\xf0\x9f\x8f\xbd', '', text)
        # emoji modifier fitzpatrick type-5
        text = re.sub('\xf0\x9f\x8f\xbe', '', text)
        # face with stuck-out tongue and winking eye
        text = re.sub('\xf0\x9f\x98\x9c', 'face_with_stuck_out_tongue_and_winking_eye_emoji ', text)
        # face with stuck-out tongue
        text = re.sub('\xf0\x9f\x98\x9b', 'face_with_stuck_out_tongue_emoji ', text)
        # crying face
        text = re.sub('\xf0\x9f\x98\xa2', 'crying_face_emoji ', text)
        # kissing face
        text = re.sub('\xf0\x9f\x98\x9a', 'kissing_face_emoji ', text)
        # grinning face
        text = re.sub('\xf0\x9f\x98\x80', 'grinning_face_emoji ', text)
        # face with tears of joy
        text = re.sub('\xf0\x9f\x98\x82', 'face_with_tears_of_joy_emoji ', text)
        # unamused face
        text = re.sub('\xf0\x9f\x98\x92', 'unamused_face_emoji ', text)
        # upside-down face
        text = re.sub('\xf0\x9f\x99\x83', 'upside_down_face_emoji ', text)
        # smiling face with heart-shaped eyes
        text = re.sub('\xf0\x9f\x98\x8d', 'smiling_face_with_heart_shaped_eyes_emoji ', text)
        # winking face
        text = re.sub('\xf0\x9f\x98\x89', 'winking_face_emoji ', text)
        # shrug emoji
        text = re.sub('\xf0\x9f\xa4\xb7', 'shrug_emoji ', text)
        # rolling on the floor laughing emoji
        text = re.sub('\xf0\x9f\xa4\xa3', 'rolling_on_the_floor_laughing_emoji ', text)
        # face palm emoji
        text = re.sub('\xf0\x9f\xa4\xa6', 'face_palm_emoji ', text)
        # smirking face
        text = re.sub('\xf0\x9f\x98\x8f', 'smirking_face_emoji ', text)
        # slightly smiling face
        text = re.sub('\xf0\x9f\x99\x82', 'slightly_smiling_face_emoji ', text)
        # smiling face with open mouth and cold sweat
        text = re.sub('\xf0\x9f\x98\x85', 'smiling_face_with_open_mouth_and_cold_sweat_emoji ', text)
        # smiling face with open mouth and smiling eyes
        text = re.sub('\xf0\x9f\x98\x84', 'smiling_face_with_open_mouth_and_smiling_eyes_emoji ', text)
        # relieved face
        text = re.sub('\xf0\x9f\x98\x8c', 'relieved_face_emoji ', text)
        # worried face
        text = re.sub('\xf0\x9f\x98\x9f', 'worried_face_emoji ', text)
        # face with ok gesture
        text = re.sub('\xf0\x9f\x99\x86', 'face_with_ok_gesture_emoji ', text)
        # face with cold sweat
        text = re.sub('\xf0\x9f\x98\x93', 'face_with_cold_sweat_emoji ', text)
        # smiling face with sunglasses
        text = re.sub('\xf0\x9f\x98\x8e', 'smiling_face_with_sunglasses_emoji ', text)
        # thinking face
        text = re.sub('\xf0\x9f\xa4\x94', 'thinking_face_emoji ', text)
        # loudly crying face
        text = re.sub('\xf0\x9f\x98\xad', 'loudly_crying_face_emoji ', text)
        # face with rolling eyes
        text = re.sub('\xf0\x9f\x99\x84', 'face_with_rolling_eyes_emoji ', text)
        # frowning face with open mouth
        text = re.sub('\xf0\x9f\x98\xa6', 'frowning_face_with_open_mouth_emoji ', text)
        # dizzy emoji
        text = re.sub('\xf0\x9f\x92\xab', 'dizzy_symbol_emoji ', text)
        # fire emoji
        text = re.sub('\xf0\x9f\x94\xa5', 'fire_emoji ', text)
        # poultry leg
        text = re.sub('\xf0\x9f\x8d\x97', 'poultry_leg_emoji ', text)
        # electric light bulb
        text = re.sub('\xf0\x9f\x92\xa1', 'electric_light_bulb_emoji ', text)
        # male sign
        text = re.sub('\xe2\x99\x82', 'male_sign ', text)
        # female sign
        text = re.sub('\xe2\x99\x80', 'female_sign ', text)

        # Handling Zero width space using https://unicode.scarfboy.com/
        # Zero width space width
        text = re.sub('\xe2\x80\x8b', '', text)
        # zero width joiner
        text = re.sub('\xe2\x80\x8d', '', text)

        # Handling special chars using https://unicode.scarfboy.com/
        # Left single quotation mark
        text = re.sub('\xe2\x80\x98', "'", text)
        # Right single quotation mark
        text = re.sub('\xe2\x80\x99', "'", text)
        # Left double quotation mark
        text = re.sub('\xe2\x80\x9c', '"', text)
        # Right double quotation mark
        text = re.sub('\xe2\x80\x9d', '"', text)
        # EM dash
        text = re.sub('\xe2\x80\x94', '-', text)
        # EN dash
        text = re.sub('\xe2\x80\x93', '-', text)
        # Fullwidth comma
        text = re.sub('\xef\xbc\x8c', ",", text)
        # Bullet
        text = re.sub('\xe2\x80\xa2', '*', text)
        # Horizontal ellipsis
        text = re.sub('\xe2\x80\xa6', '...', text)
        # checkmark to yes
        text = re.sub('\xe2\x9c\x94', 'yes', text)
        # variation selector to nothing
        text = re.sub('\xef\xb8\x8f', '', text)
        # Euro sign to Dollar sign
        text = re.sub('\xe2\x82\xac', '$', text)
        # Pound sign to Dollar sign
        text = re.sub('\xc2\xa3', '$', text)
        # Cent sign to Dollar sign
        text = re.sub('\xc2\xa2', '$', text)
        # Non-breaking space to space
        text = re.sub('\xc2\xa0', ' ', text)
        # Handling &amp
        text = text.replace("&amp;","&")
        # replace Windows-style line endings (\r\n) with Unix-style line endings
        text = re.sub('\r\n', '\n', text)
        
        # Handling acute sign using https://unicode.scarfboy.com/
        # Small letter o with acute (ó)
        text = re.sub('\xc3\xb3', "o", text)
        # Small letter e with grave (è)
        text = re.sub('\xc3\xa8', "e", text)
        # Small letter e with grave (é)
        text = re.sub('\xc3\xa9', "e", text)
        # Small letter a with grave (à)
        text = re.sub('\xc3\xa0', "a", text)
        # Acute to '
        text = re.sub('\xc2\xb4', "'", text)
        # Latin small letter dotless I used as |
        text = re.sub('\xc4\xb1', '|', text)
        return text.strip()
    
    def data_engineering(self, pdf:pd.DataFrame):
        pdf.drop("pros" , axis=1 , inplace=True)
        pdf.drop("cons" , axis=1 , inplace=True)
        pdf = pdf.rename(columns={"org_pros": "pros", "org_cons": "cons"})[['pros', 'cons', 'overall-ratings']]
        pdf["pros"] = pdf["pros"].apply(self.__clean_text)
        pdf["cons"] = pdf["cons"].apply(self.__clean_text)
        return pdf
    
    def save_generated_reviews(self, pdf:pd.DataFrame):
        pdf.to_csv(self.config.processed_data_path , index=False)