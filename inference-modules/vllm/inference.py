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

from schemas import InferenceConfig
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
from vllm.entrypoints.harmony_utils import parse_chat_output

from llm_jp_eval.cli import setup_cli
from llm_jp_eval.schemas import DatasetProfile
from llm_jp_eval_inference.generator import GeneratorBase

logger = logging.getLogger(__name__)

# Set transformers verbosity based on command
if "get_run_name" not in sys.argv:
    transformers.logging.set_verbosity_info()
else:
    transformers.logging.set_verbosity_error()


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

    def load_model(self, dataset_profile: Dict[str, DatasetProfile]):
        self.model: vllm.LLM = vllm.LLM(**self.cfg.model.model_dump())
        self.model.set_tokenizer(self.tokenizer)
        if hasattr(self.cfg.model, "reasoning_parser"):
            parser_class = ReasoningParserManager.get_reasoning_parser(self.cfg.model.reasoning_parser)
            self.reasoning_parser = parser_class(tokenizer=self.tokenizer)

    def _parse_reasoning_content(
        self, output_token_ids: list[int], output_text: str
    ) -> tuple[str | None, str]:
        # GPT-OSS: reasoning parser's extract_reasoning_content is not implemented for non-streaming mode
        # Use parse_chat_output directly to parse Harmony format token IDs
        if self.cfg.model.reasoning_parser == "openai_gptoss":
            reasoning_content, final_content, _ = parse_chat_output(output_token_ids)

            # If generation stopped during reasoning phase, use reasoning_content as final answer
            if final_content is None and reasoning_content:
                final_content = reasoning_content
                reasoning_content = None

            return reasoning_content, final_content or ""
        else:
            return self.reasoning_parser.extract_reasoning_content(
                output_text, request=ChatCompletionRequest(messages=[])
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

        if self.cfg.generation_config.max_tokens is None:
            sampling_params.max_tokens = max_output_len
        with torch.inference_mode():
            results = copy.deepcopy(target_data)
            # prompt_tokens: List[List[int]]
            outputs = self.model.generate(
                sampling_params=sampling_params, prompts=[{"prompt_token_ids": tokens} for tokens in prompt_tokens]
            )
            for i, output in enumerate(outputs):
                if hasattr(self.cfg.model, "reasoning_parser"):
                    reasoning_content, content = self._parse_reasoning_content(
                        output.outputs[0].token_ids, output.outputs[0].text
                    )
                    results["samples"][i]["reasoning_content"] = (
                        reasoning_content.lstrip() if reasoning_content else None
                    )
                    results["samples"][i]["generated"] = content.lstrip() if content else ""
                else:
                    results["samples"][i]["generated"] = output.outputs[0].text
        return results


if __name__ == "__main__":
    cfg = setup_cli(
        InferenceConfig,
        commands={
            "inference": inference,
            "get_run_name": get_run_name,
        },
    )
