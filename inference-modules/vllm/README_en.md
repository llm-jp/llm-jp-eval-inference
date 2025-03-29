# offline inference with vllm
[ english | [**日本語**](./readme.md) ]

this outputs inference results for prompts using [vllm](https://github.com/vllm-project/vllm).
for preparation before inference execution and how to run evaluations, please refer to [inference and evaluation execution methods](../../readme.md#inference-and-evaluation-execution-methods).

## installation instructions

the typical installation method using uv is as follows:

```bash
$ inference-modules/vllm
$ uv sync
```

## editing the config

please edit and use `config_sample.yaml`.

## running inference

if you want to follow all settings in `config_sample.yaml`, use this:

```bash
$ uv run python inference.py --config config_sample.yaml
```

if you want to override various settings via command line:
(the inference results will be saved to `$output_base_dir/$run_name/`)

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

example of using [eval.sh](./eval.sh) which allows simple inference execution by specifying key parameters:

- 1st arg: model_dir
- 2nd arg: tp
- 3rd arg: max_seq_len
- 4th arg: prefix_caching (enable or disable)
- 5th and subsequent args: options

```bash
$ cuda_visible_devices=0,1 ./eval.sh llm-jp/llm-jp-3-13b 2 2048 enable --run_name="13b-vllm-test1"
```
