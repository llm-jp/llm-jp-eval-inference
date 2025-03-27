import ast
import copy
import logging
import os

from typing import Any, Dict

import tensorrt_llm
import tensorrt_llm.profiler
import torch

from schemas import InferenceConfig
from tensorrt_llm.runtime import ModelRunnerCpp
from utils import load_tokenizer, read_model_name, supports_inflight_batching

from llm_jp_eval.cli import setup_cli
from llm_jp_eval.schemas import DatasetProfile
from llm_jp_eval_inference.generator import GeneratorBase

logger = logging.getLogger(__name__)


def inference(cfg: InferenceConfig):
    generator = TRTLLMGenerator(cfg)
    generator.main()


def get_run_name(cfg: InferenceConfig):
    generator = TRTLLMGenerator(cfg)
    run_name = cfg.run_name or generator._get_default_run_name()
    print(run_name)


class TRTLLMGenerator(GeneratorBase[InferenceConfig]):
    def __init__(self, cfg: InferenceConfig):
        super().__init__(cfg, "trtllm", tensorrt_llm)
        self.is_master_process = tensorrt_llm.mpi_rank() == 0
        self.model_arch, self.model_version = read_model_name(cfg.runner.engine_dir)
        self.model_name = cfg.runner.engine_dir
        self.base_model_name = (cfg.meta.base_model_name or cfg.runner.engine_dir).strip(" \t\r\n./").replace("/", "--")
        self.quantization = {
            "bfloat16": "bfloat16",
            "fp8": "fp8-W8A8",
            "sq-0.5": "int-8-W8A8",
            "awq-int4": "int4-AWQ",
        }.get(cfg.meta.quantization, cfg.meta.quantization)
        self.max_len = cfg.runner.max_seq_len
        self.tp = cfg.runner.tp
        self.pp = cfg.runner.pp
        self.cfg = cfg

        assert cfg.runner.num_beams == 1
        is_enc_dec = {
            name
            for name in os.listdir(cfg.runner.engine_dir)
            if os.path.isdir(os.path.join(cfg.runner.engine_dir, name))
        } == {"encoder", "decoder"}
        assert not is_enc_dec
        assert supports_inflight_batching(cfg.runner.engine_dir), (
            "The given engine does not support in-flight batching, fallback to python session"
        )
        if hasattr(cfg.runner, "medusa_choices"):
            assert cfg.generation_config.temperature == 1.0, "Medusa should use temperature == 1.0"

        self.run_options = []
        if cfg.runner.kv_cache_enable_block_reuse:
            self.run_options.append("kvcbr")

    def load_tokenizer(self):
        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=self.cfg.runner.tokenizer_dir,
            vocab_file=self.cfg.runner.vocab_file,
            model_name=self.model_arch,
            model_version=self.model_version,
            tokenizer_type=self.cfg.runner.tokenizer_type,
        )
        if self.pad_id is None:
            self.pad_id = self.tokenizer.pad_token_id
        if self.cfg.runner.end_id is not None:
            self.end_id = self.cfg.runner.end_id
        if self.cfg.runner.stop_words:
            self.stop_words_list = tensorrt_llm.runtime.decode_words_list(self.cfg.runner.stop_words, self.tokenizer)
        else:
            self.stop_words_list = None
        if self.cfg.runner.bad_words:
            self.bad_words_list = tensorrt_llm.runtime.decode_words_list(self.cfg.runner.bad_words, self.tokenizer)
        else:
            self.bad_words_list = None

    def load_model(self, dataset_profile: Dict[str, DatasetProfile]):
        runner_kwargs = self.cfg.runner
        args = dict(
            engine_dir=runner_kwargs.engine_dir,
            lora_dir=runner_kwargs.lora_dir,
            rank=tensorrt_llm.mpi_rank(),
            debug_mode=runner_kwargs.debug_mode,
            lora_ckpt_source=runner_kwargs.lora_ckpt_source,
            gpu_weights_percent=runner_kwargs.gpu_weights_percent,
            is_enc_dec=False,
            max_beam_width=runner_kwargs.num_beams,
            max_attention_window_size=runner_kwargs.max_attention_window_size,
            sink_token_length=runner_kwargs.sink_token_length,
            max_tokens_in_paged_kv_cache=runner_kwargs.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=runner_kwargs.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=runner_kwargs.kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=runner_kwargs.enable_chunked_context,
            multi_block_mode=runner_kwargs.multi_block_mode,
            enable_context_fmha_fp32_acc=runner_kwargs.enable_context_fmha_fp32_acc,
            max_input_len=dataset_profile["(total)"].max_input_len,
        )
        if runner_kwargs.medusa_choices is not None:
            medusa_choices = ast.literal_eval(runner_kwargs.medusa_choices)
            args.update(medusa_choices=medusa_choices)
        if self.is_master_process:
            logger.info(f"runner_kwargs: {args}")
        self.runner = ModelRunnerCpp.from_dir(**args)

    def generate(
        self,
        max_input_len: int,
        max_output_len: int,
        target_data: Dict[str, Any],
        prompt_tokens: list,
        prompt_lengths: list,
    ) -> Dict[str, Any]:
        self.runner.max_input_len = max_input_len
        if max_output_len > self.runner.max_seq_len - self.runner.max_input_len:
            max_new_tokens = self.runner.max_seq_len - self.runner.max_input_len
            if self.is_master_process:
                logger.info(
                    f"{max_output_len=} of {target_data['target_dataset']} exceeds {self.runner.max_seq_len=} - {self.runner.max_input_len=}"
                )
        else:
            max_new_tokens = max_output_len
        results = copy.deepcopy(target_data)
        max_batch_size = self.runner.max_batch_size
        for batch_begin in range(0, len(prompt_tokens), max_batch_size):
            batch_prompt_tokens = [
                torch.tensor(_, dtype=torch.int32) for _ in prompt_tokens[batch_begin : batch_begin + max_batch_size]
            ]
            batch_prompt_lengths = prompt_lengths[batch_begin : batch_begin + max_batch_size]
            with torch.inference_mode():
                outputs = self.runner.generate(
                    batch_input_ids=batch_prompt_tokens,
                    encoder_input_ids=None,
                    max_new_tokens=max_new_tokens,
                    end_id=self.end_id,
                    pad_id=self.pad_id,
                    stop_words_list=self.stop_words_list,
                    bad_words_list=self.bad_words_list,
                    output_cum_log_probs=False,
                    output_log_probs=False,
                    prompt_table=self.cfg.runner.prompt_table_path,
                    streaming=self.cfg.runner.streaming,
                    prompt_tasks=self.cfg.runner.prompt_tasks,
                    medusa_choices=self.cfg.runner.medusa_choices,
                    return_all_generated_tokens=self.cfg.runner.return_all_generated_tokens,
                    output_sequence_lengths=True,
                    return_dict=True,
                    **self.cfg.generation_config.model_dump(exclude={"max_new_tokens"}),
                )
                torch.cuda.synchronize()
            if self.is_master_process:
                output_ids = outputs["output_ids"]
                sequence_lengths = outputs["sequence_lengths"]
                for batch_idx in range(len(batch_prompt_tokens)):
                    output_begin = batch_prompt_lengths[batch_idx]
                    output_end = sequence_lengths[batch_idx][0]
                    generated_ids = output_ids[batch_idx][0][output_begin:output_end].tolist()
                    generated_text = self.tokenizer.decode(generated_ids)
                    results["samples"][batch_begin + batch_idx]["generated"] = generated_text
        return results


if __name__ == "__main__":
    cfg = setup_cli(
        InferenceConfig,
        commands={
            "inference": inference,
            "get_run_name": get_run_name,
        },
    )
