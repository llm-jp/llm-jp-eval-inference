# Hugging Face Transformers (TextGenerationPipeline) によるoffline推論
[ [**English**](./README_en.md) | 日本語 ]

Transformersの[pipeline](https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/pipelines/text_generation.py#L38)を用いてプロンプトの推論結果を出力します。

推論実行前の準備や評価の実行方法については [推論および評価実行方法](../../README.md#推論および評価実行方法) を参照してください。

## インストール手順

uvによる典型的なインストール方法は次の通りです。

```bash
$ uv sync
```

## configの編集

`config_sample.yaml`を編集して使用してください。

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
  --model.pretrained_model_name_or_path=$MODEL_NAME \
  --tokenizer.pretrained_model_name_or_path=$MODEL_NAME \
  --run_name="$RUN_NAME" \
  --output_base_dir="$OUTPUT_BASE_DIR" \
  --prompt_json_path=['../../../local_files/datasets/2.0.0/evaluation/path/to/prompts/*.eval-prompt.json']
```

主要パラメータを指定して簡易に推論を実行できる [eval.sh](./eval.sh) の使用例。

- 第1引数: MODEL_DIR
- 第2引数: BATCH_SIZE
- 第3引数: MAX_SEQ_LEN
- 第4引数以降: OPTIONS

```bash
$ ./eval.sh llm-jp/llm-jp-3-13b 3 2048 --run_name="13b-transformers-test1"
```
