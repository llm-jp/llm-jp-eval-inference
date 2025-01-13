from pathlib import Path

from pydantic import BaseModel, Field, DirectoryPath

class WandBConfig(BaseModel):
    launch: bool = Field(False, description="true for WANDB Launch. notice: if it is true, all other configurations will be overwrited by launch config")
    log: bool = Field(False, description="true for logging WANDB in evaluate_llm.py")
    entity: str = Field("your/WANDB/entity")
    project: str = Field("your/WANDB/project")
    run_name: str | None = Field(None, description="default run_name is f'{model.pretrained_model_name_or_path}_yyyyMMdd_hhmmss'")

class BaseInferenceConfig(BaseModel):
    prompt_json_path: str | list[str] = Field(["../../../dataset/1.4.1/evaluation/test/prompts/*.eval-prompt.json"], description="path to the prompt json file")
    output_base_dir: Path = Field("output", description="output directory")
    # If null, the top level run_name or the default run_name determined by model configuration will be used. 
    # Do not specify both this value and top level run_name at the same time. The final output path is {output_base_dir}/{run_name}/
    run_name : str | None = Field(None, description="run_name for logging")