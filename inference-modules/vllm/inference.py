import copy
import logging
import os
import sys

from typing import Any, Dict

# Suppress vLLM logs before importing (for get_run_name command)
if "get_run_name" in sys.argv:
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)

import torch
import transformers
import vllm
import vllm.sampling_params

from output_parsing import ParsedOutput
from parser_adapter import ParserAdapter
from schemas import InferenceConfig
from transformers import PreTrainedTokenizerBase
from vllm.reasoning import ReasoningParserManager
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.tool_parsers import ToolParserManager

from llm_jp_eval.cli import setup_cli
from llm_jp_eval.schemas import DatasetProfile
from llm_jp_eval_inference.generator import GeneratorBase

logger = logging.getLogger(__name__)

# Set transformers verbosity based on command
if "get_run_name" not in sys.argv:
    transformers.logging.set_verbosity_info()
else:
    transformers.logging.set_verbosity_error()


def _parser_name(value: str | None) -> str | None:
    # Config templates write an unset parser as "" or "null"
    if value is None or value.strip().lower() in ("", "null"):
        return None
    return value.strip()


def inference(cfg: InferenceConfig):
    generator = VLLMGenerator(cfg)
    generator.main()


def get_run_name(cfg: InferenceConfig):
    generator = VLLMGenerator(cfg)
    run_name = cfg.run_name or generator._get_default_run_name()
    print(run_name)


class VLLMGenerator(GeneratorBase[InferenceConfig]):
    def __init__(self, cfg: InferenceConfig):
        super().__init__(cfg, "vllm", vllm)
        self.model_name = cfg.model.model
        if cfg.meta.base_model_name:
            self.base_model_name = cfg.meta.base_model_name
        else:
            model_name_replaced = self.model_name.strip(" \t\r\n./").replace("/", "--")
            for suffix, quantization in {
                "-FP8": "fp8-W8A8",
                "-quantized.w8a8": "int-8-W8A8",
                "-AWQ-INT4": "int4-AWQ",
                "-GPTQ-INT4": "int4-GPTQ",
            }.items():
                if model_name_replaced.endswith(suffix):
                    self.base_model_name = model_name_replaced[: -len(suffix)]
                    self.quantization = quantization
                    break
            else:
                self.base_model_name = model_name_replaced
                self.quantization = cfg.model.dtype
        if cfg.meta.quantization:
            self.quantization = cfg.meta.quantization
        self.max_len = cfg.model.max_model_len
        self.tp = cfg.model.tensor_parallel_size
        self.pp = cfg.model.pipeline_parallel_size
        self.reasoning_parser = _parser_name(cfg.model.reasoning_parser)
        self.parser_adapter: ParserAdapter | None = None

        self.run_options = []
        if cfg.model.enable_prefix_caching:
            self.run_options.append("apc")
        if cfg.model.gpu_memory_utilization is not None:
            self.run_options.append("gmu" + f"{cfg.model.gpu_memory_utilization:.3f}"[1:])

    def load_tokenizer(self):
        self.tokenizer: PreTrainedTokenizerBase = transformers.AutoTokenizer.from_pretrained(
            **self.cfg.tokenizer.model_dump()
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.is_master_process:
                logger.info("setting pad_token_id=eos_token_id")

    def supports_tool_calls(self) -> bool:
        return _parser_name(self.cfg.tool_call_parser) is not None

    def load_model(self, dataset_profile: Dict[str, DatasetProfile]):
        tool_call_parser = _parser_name(self.cfg.tool_call_parser)
        # As with vllm serve, a plugin parser takes precedence over the built-in one of the same name
        if self.reasoning_parser is not None and self.cfg.model.reasoning_parser_plugin:
            ReasoningParserManager.import_reasoning_parser(self.cfg.model.reasoning_parser_plugin)
        if tool_call_parser is not None and self.cfg.tool_parser_plugin:
            ToolParserManager.import_tool_parser(self.cfg.tool_parser_plugin)
        if self.reasoning_parser is not None:
            self.cfg.model.reasoning_parser = self.reasoning_parser
            # add context length for reasoning content
            if self.cfg.reasoning_content_length is not None:
                self.cfg.model.max_model_len += self.cfg.reasoning_content_length
            else:
                # set model context length
                self.cfg.model.max_model_len = None
        self.model: vllm.LLM = vllm.LLM(**self.cfg.model.model_dump())
        # Inject the external tokenizer into vLLM's renderer (replaces deprecated set_tokenizer)
        self.model.llm_engine.renderer.tokenizer = get_cached_tokenizer(self.tokenizer)
        if self.reasoning_parser is not None or tool_call_parser is not None:
            self.parser_adapter = ParserAdapter(
                self.tokenizer,
                self.reasoning_parser,
                tool_call_parser,
                # Same check as the OpenAI-compatible server
                is_harmony=self.model.model_config.hf_config.model_type == "gpt_oss",
                chat_template_kwargs=self.cfg.tokenize_kwargs,
            )

    def generate(
        self,
        max_input_len: int,
        max_output_len: int,
        target_data: Dict[str, Any],
        prompt_tokens: list,
        prompt_lengths: list,
    ) -> Dict[str, Any]:
        sampling_params = vllm.sampling_params.SamplingParams(**self.cfg.generation_config.model_dump())
        if self.parser_adapter is not None:
            self.parser_adapter.adjust_sampling_params(sampling_params, target_data.get("tools"))

        if self.cfg.generation_config.max_tokens is None:
            # Add reasoning_content_length for models with reasoning_parser
            if self.reasoning_parser is not None:
                if self.cfg.reasoning_content_length is None:
                    sampling_params.max_tokens = None
                else:
                    logger.info(f"Adding reasoning content length: {self.cfg.reasoning_content_length}")
                    sampling_params.max_tokens = max_output_len + self.cfg.reasoning_content_length
            else:
                sampling_params.max_tokens = max_output_len
            logger.info(f"Setting max_tokens to {sampling_params.max_tokens}")

        with torch.inference_mode():
            results = copy.deepcopy(target_data)
            # prompt_tokens: List[List[int]]
            outputs = self.model.generate(
                sampling_params=sampling_params, prompts=[{"prompt_token_ids": tokens} for tokens in prompt_tokens]
            )
            for sample, output in zip(results["samples"], outputs, strict=True):
                tools = sample.get("tools") or target_data.get("tools")
                parsed = (
                    self.parser_adapter.parse(output.outputs[0].text, list(output.outputs[0].token_ids), tools)
                    if self.parser_adapter is not None
                    else ParsedOutput(content=output.outputs[0].text)
                )
                sample.update(parsed.to_result_fields(self.reasoning_parser is not None, bool(tools)))
        return results


if __name__ == "__main__":
    cfg = setup_cli(
        InferenceConfig,
        commands={
            "inference": inference,
            "get_run_name": get_run_name,
        },
    )
