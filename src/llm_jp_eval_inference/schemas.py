from pathlib import Path

from pydantic import BaseModel, Field

from llm_jp_eval.schemas import BaseInferenceResultConfig, InferenceResultInfo

# Define the set of allowed keyword arguments for the tokenizer's tokenize method
# apply_chat_template
# https://github.com/huggingface/transformers/blob/v5.0.0rc1/src/transformers/tokenization_utils_base.py#L2523-L2541
tokenize_kwargs: set = {}


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
    meta: MetaInfo = Field(default_factory=MetaInfo)
    inference_result_info: InferenceResultInfo | None = None
    tokenize_kwargs: dict = Field(
        {"add_special_tokens": True},
        description="additional arguments for tokenizer.tokenize",
    )
    apply_chat_template: bool = Field(False, description="apply chat template to the prompt")

    prompt_json_path: str | list[str] = Field(
        ["../../../dataset/1.4.1/evaluation/test/prompts/*.eval-prompt.json"],
        description="path to the prompt json file",
    )
    output_base_dir: Path = Field(Path("outputs"), description="output directory")
