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
| `model.tool_call_parser` | vLLM tool parser name (e.g. `hermes`, `qwen3_coder`). Required for datasets with tools; `tool_calls` are read from the generated text |
| `model.tool_parser_plugin` | Module name or path to a Python file that registers a custom tool parser. Refer to its registered name with `tool_call_parser` |

The parser settings correspond to `--reasoning-parser` / `--reasoning-parser-plugin` / `--tool-call-parser` / `--tool-parser-plugin` of the vLLM server, and reasoning and tool calls are extracted with the same parser as the server.
The reasoning parser and the tool parser are set separately, and both must match the output format of the model. The tool call format can change between generations of the same model family, so the reasoning parser does not determine the tool parser.
A tool parser that does not match the format raises no error; tool calls are just not read and the samples are scored as such, so use the parser given in the model card.

| Model | `reasoning_parser` | `tool_call_parser` | Source |
| --- | --- | --- | --- |
| Qwen3 | `qwen3` | `hermes` | [Qwen documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html#vllm) |
| Qwen3.5 | `qwen3` | `qwen3_coder` | [Model card](https://huggingface.co/Qwen/Qwen3.5-2B) |
| Gemma 4 | `gemma4` | `gemma4` | [vLLM recipe](https://github.com/vllm-project/recipes/blob/main/Google/Gemma4.md) |
| LLM-jp-4 | `llmjp4` | `llmjp4` | [llm-jp-vllm](https://github.com/llm-jp/llm-jp-vllm) |

The `llmjp4` parser for LLM-jp-4 is registered at startup from llm-jp-vllm, which is a dependency, so it can be selected by name alone without a plugin.

```yaml
apply_chat_template: true
model:
  model: Qwen/Qwen3.5-2B
  reasoning_parser: qwen3
  tool_call_parser: qwen3_coder
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
