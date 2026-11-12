# vLLMによるoffline推論
[ [**English**](./README_en.md) | 日本語 ]

[vLLM](https://github.com/vllm-project/vllm)を用いてプロンプトの推論結果を出力します。
推論実行前の準備や評価の実行方法については [推論および評価実行方法](../../README.md#推論および評価実行方法) を参照してください。

## インストール手順

uvによる典型的なインストール方法は次の通りです。

```bash
$ inference-modules/vllm
$ uv sync
```

vLLM と torch は CUDA 12.9 版を使うため、NVIDIA ドライバ 535 以降で動きます。
ドライバが 580 未満の環境では FlashInfer のサンプラーが動かないため、`VLLM_USE_FLASHINFER_SAMPLER=0` を設定して実行してください。

## configの編集

`config_sample.yaml`を編集して使用してください。

## chat template・reasoning・tool calling の設定

| 設定 | 内容 |
| --- | --- |
| `apply_chat_template` | `true` で tokenizer の chat template を使う。messages 入力のデータセット（JMultiWOZ-TC など）では必須 |
| `tokenize_kwargs` | tokenizer に渡す追加引数。chat template を使うときは template 変数としても渡る（例: `enable_thinking`） |
| `model.reasoning_parser` | vLLM の reasoning parser 名（例: `qwen3`）。指定すると思考部分を `reasoning_content` に分けて保存する |
| `model.reasoning_parser_plugin` | 独自の reasoning parser を登録するモジュール名または Python ファイルのパス（例: `llm_jp_vllm.llmjp4`） |
| `reasoning_content_length` | reasoning 用に出力長とコンテキスト長へ足すトークン数 |
| `tool_call_parser` | vLLM の tool parser 名（例: `hermes`、`llama3_json`、`mistral`）。tools のあるデータセットでは必須で、生成文から `tool_calls` を読む |
| `tool_parser_plugin` | 独自の tool parser を登録するモジュール名または Python ファイルのパス。`tool_call_parser` でその登録名を指定する |

parser の設定は vLLM サーバの `--reasoning-parser` / `--reasoning-parser-plugin` / `--tool-call-parser` / `--tool-parser-plugin` に対応し、思考部分と tool call はサーバと同じ parser で取り出します。
`tool_call_parser` と `tool_parser_plugin` は vLLM のエンジン引数ではないため、`model` ではなく最上位に書きます。
LLM-jp-4 の `llmjp4` parser は、依存に含まれる [llm-jp-vllm](https://github.com/llm-jp/llm-jp-vllm) を起動時に読み込んで登録するため、plugin を指定せずに名前だけで使えます（`reasoning_parser: llmjp4`。tool calling では `tool_call_parser: llmjp4`）。

```yaml
apply_chat_template: true
model:
  model: Qwen/Qwen3-8B
  reasoning_parser: qwen3
tool_call_parser: hermes
```

## 推論の実行

`config_sample.yaml`の設定にすべて従う場合はこちら。

```bash
$ uv run python inference.py inference --config config_sample.yaml
```

コマンドラインで各種設定をオーバーライドする場合はこちら。
（推論結果の保存先ディレクトリは `$OUTPUT_BASE_DIR/$RUN_NAME/` となる）

```bash
$ MODEL_NAME=llm-jp/llm-jp-3-13b
$ OUTPUT_BASE_DIR=./output
$ uv run python inference.py inference \
  --config ./config_sample.yaml \
  --model.model=$MODEL_NAME \
  --tokenizer.pretrained_model_name_or_path=$MODEL_NAME \
  --run_name="$RUN_NAME" \
  --output_base_dir="$OUTPUT_BASE_DIR" \
  --prompt_json_path=['../../../local_files/datasets/2.0.0/evaluation/path/to/prompts/*.eval-prompt.json']
```


主要パラメータを指定して簡易に推論を実行できる [eval.sh](./eval.sh) の使用例。

- 第1引数: MODEL_DIR
- 第2引数: TP
- 第3引数: MAX_SEQ_LEN
- 第4引数: PREFIX_CACHING (enable or disable)
- 第5引数以降: OPTIONS

```bash
$ CUDA_VISIBLE_DEVICES=0,1 ./eval.sh llm-jp/llm-jp-3-13b 2 2048 enable --run_name="13b-vllm-test1"
```
