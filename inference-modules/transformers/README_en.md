# Offline Inference with Hugging Face Transformers (TextGenerationPipeline)
[ English | [**日本語**](./README.md) ]

This outputs inference results for prompts using the Transformers [pipeline](https://github.com/huggingface/transformers/blob/v4.42.3/src/transformers/pipelines/text_generation.py#L38).

For preparation before inference execution and how to run evaluations, please refer to [Inference and Evaluation Execution Methods](../../README.md#inference-and-evaluation-execution-methods).

## Installation Instructions

The typical installation method using uv is as follows:

```bash
$ uv sync
```

## Editing the Config

Please edit and use `config_sample.yaml`.

## Running Inference

If you want to follow all settings in `config_sample.yaml`, use this:

```bash
$ uv run python inference.py --config config_sample.yaml
```

If you want to override various settings via command line:
(The inference results will be saved to `$OUTPUT_BASE_DIR/$RUN_NAME/`)

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

Example of using [eval.sh](./eval.sh) which allows simple inference execution by specifying key parameters:

- 1st arg: MODEL_DIR
- 2nd arg: BATCH_SIZE
- 3rd arg: MAX_SEQ_LEN
- 4th and subsequent args: OPTIONS

```bash
$ ./eval.sh llm-jp/llm-jp-3-13b 3 2048 --run_name="13b-transformers-test1"
```
