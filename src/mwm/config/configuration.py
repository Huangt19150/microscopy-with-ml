import os
import json
from box import ConfigBox
from mwm.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from mwm.utils.common import read_yaml, create_directories, load_json
from mwm.entity.config_entity import DataIngestionConfig


def get_params(params_filepath: str, mode_key: str) -> ConfigBox:
    params_raw = load_json(params_filepath)
    try:
        params = params_raw["common"] + params_raw[mode_key]
    except KeyError:
        raise KeyError(f"mode_key: {mode_key} not found in params.json. Please check the file.")
    params["image_size"] = params["image_size_lut"][params["network"]]
    params.pop("image_size_lut", None)

    return params

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file= config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

