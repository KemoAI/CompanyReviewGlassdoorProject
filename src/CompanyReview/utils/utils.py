import os
import yaml
import json
import joblib
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from box.exceptions import BoxValueError
from box import ConfigBox

@ensure_annotations
def read_yaml(path_to_yaml:Path)-> ConfigBox:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
        return ConfigBox(content)