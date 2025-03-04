from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataLoadingConfig:
    root_data_path:Path
    processed_data_path:Path