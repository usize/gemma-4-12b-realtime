#!/usr/bin/env bash

set -eo

llama-server \
  --model ./models/Qwen3.6-27B-Q4_K_S.gguf \
  --mmproj ./models/mmproj-F16.gguf \
  --ctx-size 128000 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --threads 12 \
  --n-gpu-layers 99 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --port 8081 \
  --host 0.0.0.0
