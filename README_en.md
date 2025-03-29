# llm-jp-eval-inference
[ English | [**日本語**](./README.md) ]

In this repository, we primarily release implementations of fast batch inference processing for [llm-jp-eval](https://github.com/llm-jp-eval/llm-jp-eval) using the following libraries:
For installation and inference execution, please refer to the README.md within each module.

- [vLLM](./llm-jp-eval-inference/vllm/)
- [TensorRT-LLM](./llm-jp-eval-inference/trtllm/)
- [Hugging Face Transformers](./llm-jp-eval-inference/transformers) (baseline)

In addition, a tool for run management using Weights & Biases are published in [wandb_run_management](./wandb_run_management).

# Inference and Evaluation Execution Methods

Please refer to the [Inference Execution Method](https://github.com/llm-jp/llm-jp-eval/blob/dev/README_en.md#inference-methods) and [Evaluation Method](https://github.com/llm-jp/llm-jp-eval/blob/dev/README_en.md#evaluation-methods) in llm-jp-eval.
