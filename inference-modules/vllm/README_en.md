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

vLLM and torch are the CUDA 12.9 builds, which run on NVIDIA driver 535 or later.
With a driver older than 580, the FlashInfer sampler does not work; set `VLLM_USE_FLASHINFER_SAMPLER=0` when running.

## editing the config

please edit and use `config_sample.yaml`.

## Chat Template, Reasoning and Tool Calling

| Setting | Description |
| --- | --- |
| `apply_chat_template` | `true` uses the chat template of the tokenizer. Required for datasets with messages input such as JMultiWOZ-TC |
| `tokenize_kwargs` | Extra arguments for the tokenizer. With a chat template they are also passed as template variables (e.g. `enable_thinking`) |
| `model.reasoning_parser` | vLLM reasoning parser name (e.g. `qwen3`). The reasoning part is saved separately as `reasoning_content` |
| `model.reasoning_parser_plugin` | Module name or path to a Python file that registers a custom reasoning parser (e.g. `llm_jp_vllm.llmjp4`) |
| `reasoning_content_length` | Tokens added to the output length and context length for reasoning |
| `tool_call_parser` | vLLM tool parser name (e.g. `hermes`, `llama3_json`, `mistral`). Required for datasets with tools; `tool_calls` are read from the generated text |
| `tool_parser_plugin` | Module name or path to a Python file that registers a custom tool parser. Refer to its registered name with `tool_call_parser` |

The parser settings correspond to `--reasoning-parser` / `--reasoning-parser-plugin` / `--tool-call-parser` / `--tool-parser-plugin` of the vLLM server, and reasoning and tool calls are extracted with the same parser as the server.
`tool_call_parser` and `tool_parser_plugin` are not engine arguments, so write them at the top level instead of under `model`.
The `llmjp4` parser for LLM-jp-4 is registered at startup from [llm-jp-vllm](https://github.com/llm-jp/llm-jp-vllm), which is a dependency, so it can be selected by name alone without a plugin (`reasoning_parser: llmjp4`; for tool calling, `tool_call_parser: llmjp4`).

```yaml
apply_chat_template: true
model:
  model: Qwen/Qwen3-8B
  reasoning_parser: qwen3
tool_call_parser: hermes
```

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
