from typing import Literal
from pydantic import BaseModel
from transformers import GenerationConfig
from llm_jp_eval_inference.schemas import BaseInferenceConfig, WandBConfig

class ModelConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False

class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    use_fast: bool = True
    padding_side: Literal["left", "right"] = "left"
    model_max_length: int = 2048
    add_special_tokens: bool = False

# these values will be used only for logging and wandb (no effect for running environment)
class MetaInfo(BaseModel):
  quantization: str | None = None
  run_name_lib_type: str | None = None
  run_name_gpu_type: str | None = None
  run_name_suffix: str | None = None
  model_conversion_time: float = 0.0

class InferenceConfig(BaseModel):
    inference_config: BaseInferenceConfig
    wandb: WandBConfig

    # transformers specific configurations
    run: MetaInfo
    model: ModelConfig
    tokenizer: TokenizerConfig
    generation_config: GenerationConfig

