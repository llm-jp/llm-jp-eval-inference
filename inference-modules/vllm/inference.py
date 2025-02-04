import copy
import logging
from typing import Any, Dict

import torch
import transformers
import vllm
import vllm.sampling_params
from llm_jp_eval.cli import setup_cli
from schemas import InferenceConfig
from transformers import PreTrainedTokenizerBase

from llm_jp_eval_inference.generator import GeneratorBase
from llm_jp_eval_inference.schemas import DatasetProfile

logger = logging.getLogger(__name__)
transformers.logging.set_verbosity_info()


def main():
    cfg = setup_cli(InferenceConfig)
    generator = VLLMGenerator(cfg)
    generator.main()


class VLLMGenerator(GeneratorBase[InferenceConfig]):
    def __init__(self, cfg: InferenceConfig):
        super().__init__(cfg, "vllm", vllm)
        self.model_name = cfg.model.model
        if cfg.run.base_model_name:
            self.base_model_name = cfg.run.base_model_name
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
        if cfg.run.quantization:
            self.quantization = cfg.run.quantization
        self.max_len = cfg.model.max_model_len
        self.tp = cfg.model.tensor_parallel_size
        self.pp = cfg.model.pipeline_parallel_size

        self.run_options = []
        if cfg.model.enable_prefix_caching:
            self.run_options.append("apc")
        if "gpu_memory_utilization" in cfg.model and cfg.model.gpu_memory_utilization != 0.9:
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

    def generate(
        self,
        max_input_len: int,
        max_output_len: int,
        target_data: Dict[str, Any],
        prompt_tokens: list,
        prompt_lengths: list,
    ) -> Dict[str, Any]:
        sampling_params = vllm.sampling_params.SamplingParams(**self.cfg.generation_config.model_dump())
        if "max_tokens" not in self.cfg.generation_config:
            sampling_params.max_tokens = max_output_len
        with torch.inference_mode():
            results = copy.deepcopy(target_data)
            outputs = self.model.generate(sampling_params=sampling_params, prompt_token_ids=prompt_tokens)
            for i, output in enumerate(outputs):
                results["samples"][i]["generated"] = output.outputs[0].text
        return results


if __name__ == "__main__":
    main()
