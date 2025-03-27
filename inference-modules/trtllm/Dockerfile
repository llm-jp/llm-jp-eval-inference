FROM nvcr.io/nvidia/tritonserver:24.08-trtllm-python-py3

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Install System Python environment
ENV UV_PROJECT_ENVIRONMENT=/usr/

RUN apt update -y && \
    apt install -y uuid-runtime git-lfs && \
    git lfs install

RUN mkdir -p /workspace/llm-jp-eval-inference/inference-modules/trtllm
WORKDIR /workspace/llm-jp-eval-inference/inference-modules/trtllm

COPY ./inference-modules/trtllm/pyproject.toml .
COPY ./pyproject.toml README.md /workspace/llm-jp-eval-inference/
COPY ./src /workspace/llm-jp-eval-inference/src/

# Apply workaround for trust_remote_code issue
RUN sed -i -r "s/split=split,$/split=split, trust_remote_code=trust_remote_code,/" /usr/local/lib/python3.10/dist-packages/tensorrt_llm/models/convert_utils.py

RUN cd /workspace/llm-jp-eval-inference/inference-modules/trtllm && \
    uv sync && pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com

