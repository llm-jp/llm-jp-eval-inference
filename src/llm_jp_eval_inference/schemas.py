from pathlib import Path

from llm_jp_eval.schemas import BaseInferenceResultConfig
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    num_prompt: int
    max_input_len: int
    sum_input_len: int
    max_output_len: int
    max_seq_len: int
    sum_seq_len: int


class InferenceResultInfo(BaseModel):
    dump_prompts_config: dict
    dataset_profiles: dict[str, DatasetProfile]
    benchmark: dict
    time_profile: dict


# these values will be used only for logging and wandb (no effect for running environment)
class MetaInfo(BaseModel):
    base_model_name: str | None = None
    quantization: str | None = None
    lib_type: str | None = None
    gpu_type: str | None = None
    run_name_suffix: str | None = None
    model_conversion_time: float = 0.0


# Inherit the interface for passing to llm-jp-eval
class BaseInferenceConfig(BaseInferenceResultConfig):
    run: MetaInfo = Field(default_factory=MetaInfo)
    inference_result_info: InferenceResultInfo | None = None
    tokenize_kwargs: dict = Field(
        {"add_special_tokens": True},
        description="additional arguments for tokenizer.tokenize",
    )

    prompt_json_path: str | list[str] = Field(
        ["../../../dataset/1.4.1/evaluation/test/prompts/*.eval-prompt.json"],
        description="path to the prompt json file",
    )
    output_base_dir: Path = Field(Path("outputs"), description="output directory")
