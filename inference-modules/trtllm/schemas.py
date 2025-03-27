from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from llm_jp_eval_inference.schemas import BaseInferenceConfig


class RunnerConfig(BaseModel):
    model_config = ConfigDict(extra='allow')
    engine_dir: str
    tokenizer_dir: str
    enable_chunked_context: bool = True
    enable_context_fmha_fp32_acc: bool = True
    kv_cache_enable_block_reuse: bool = True
    return_all_generated_tokens: bool = False
    max_seq_len: int = 2048
    tp: int = 1
    pp: int = 1
    num_beams: int = 1
    gpu_weights_percent: float = 1.0

    vocab_file: str | None = None
    tokenizer_type: str | None = None
    end_id: int | None = None
    stop_words: list[str] | None = None
    bad_words: list[str] | None = None
    lora_dir: str | None = None
    debug_mode: bool = False
    lora_ckpt_source: str | None = None
    max_attention_window_size: int | None = None
    sink_token_length: int | None = None
    max_tokens_in_paged_kv_cache: int | None = None
    kv_cache_free_gpu_memory_fraction: float | None = None
    multi_block_mode: bool = False
    medusa_choices: str | None = None
    lora_task_uids: list[str] | None = None
    prompt_table_path: str | None = None
    prompt_tasks: list[str] | None = None
    streaming: bool = False

class GenerationConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    max_new_tokens: int = 256
    do_sample: bool = True
    num_return_sequences: int = 1
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = False
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    no_repeat_ngram_size: int | None = None

    random_seed: int | None = None
    lora_uids: list[str] | None = None

class TokenizerConfig(BaseModel):
    pretrained_model_name_or_path: str
    trust_remote_code: bool = False
    use_fast: bool = True
    padding_side: Literal["left", "right"] = "left"
    model_max_length: int = 2048


class InferenceConfig(BaseInferenceConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    runner: RunnerConfig

    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
