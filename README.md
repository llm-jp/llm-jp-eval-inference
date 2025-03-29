# llm-jp-eval-inference
[ [**English**](./README_en.md) | 日本語 ]

本リポジトリでは主に、[llm-jp-eval](https://github.com/llm-jp-eval/llm-jp-eval)向けに、次のライブラリを用いた高速なバッチ推論処理を実装を公開します。

- [vLLM](./inference-modules/vllm/)
- [TensorRT-LLM](./inference-modules/trtllm/)
- [Hugging Face Transformers](./inference-modules/transformers) (baseline)

また、[Weights & Biases Run管理ツール](./src//wandb_run_management)の実装を公開します。

インストール、推論の実行についてはそれぞれのmodule内README.mdを参照してください。


# 推論および評価実行方法

llm-jp-evalにおける[推論実行方法](https://github.com/llm-jp/llm-jp-eval?tab=readme-ov-file#推論実行方法)および[評価方法](https://github.com/llm-jp/llm-jp-eval?tab=readme-ov-file#評価方法)を参考にしてください。
