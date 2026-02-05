# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE="astral/uv:debian-slim"
# The latest main will be used if arg unspecified
ARG VLLM_COMMIT_HASH=""

FROM $BASE_IMAGE

RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt install --yes git 
RUN \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt install --yes build-essential

RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --seed --clear -p "3.12.12" "/opt/venv"

ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

# Build vLLM
WORKDIR /workspace/vllm
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_COMMIT_HASH

RUN git clone $VLLM_REPO . && \
    if [ -n "$VLLM_COMMIT_HASH" ]; then \
        git checkout $VLLM_COMMIT_HASH; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    VLLM_TARGET_DEVICE=tpu uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    -e .[bench]

# Install tpu_inference
WORKDIR /workspace/tpu_inference

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -e .

CMD ["/bin/bash"]
