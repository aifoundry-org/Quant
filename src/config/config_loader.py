import yaml
from pydantic import ValidationError
from src.config.config_schema import ConfigSchema

def load_and_validate_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    try:
        validated_config = ConfigSchema.model_validate(config)
    except ValidationError as e:
        print("Config validation error:", e)
        raise
    
    return validated_config