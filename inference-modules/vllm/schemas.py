from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from llm_jp_eval_inference.schemas import BaseInferenceConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int = 2048
    num_scheduler_steps: int = 8
    enable_prefix_caching: bool = True


class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    use_fast: bool = True
    padding_side: Literal["left", "right"] = "left"


# https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py#L87
class VLLMSamplingParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = None
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None


class InferenceConfig(BaseInferenceConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # transformers specific configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    generation_config: VLLMSamplingParams = Field(default_factory=VLLMSamplingParams)
