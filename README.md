# llm-jp-eval-inference
[ [**English**](./README_en.md) | 日本語 ]

本リポジトリでは主に、[llm-jp-eval](https://github.com/e-mon/llm-jp-eval-inference)向けに、次のライブラリを用いた高速なバッチ推論処理を実装を公開します。

- [vLLM](./llm-jp-eval-inference/vllm/)
- [TensorRT-LLM](./llm-jp-eval-inference/trtllm/)
- [Hugging Face Transformers](./llm-jp-eval-inference/transformers) (baseline)

また、[Weights & Biases Run管理ツール](./llm-jp-eval-inference/wandb_run_management)の実装を公開します。

インストール、推論の実行についてはそれぞれのmodule内README.mdを参照してください。
