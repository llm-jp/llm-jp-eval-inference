from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from transformers import GenerationConfig

from llm_jp_eval_inference.schemas import BaseInferenceConfig


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


class InferenceConfig(BaseInferenceConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # transformers specific configurations
    model: ModelConfig
    tokenizer: TokenizerConfig
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    pipeline_kwargs: dict = Field(default_factory=dict)

    @field_validator("generation_config", mode="before")
    @classmethod
    def validate_generation_config(cls, value: Any) -> GenerationConfig:
        if isinstance(value, dict):
            return GenerationConfig(**value)
        return value

    @field_serializer("generation_config")
    def serialize_generation_config(self, config: GenerationConfig) -> dict:
        return config.to_dict()
