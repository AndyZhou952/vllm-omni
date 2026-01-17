#!/bin/bash
# Online diffusion serving with vLLM-Omni (OpenAI-compatible API).

MODEL="${MODEL:-stabilityai/stable-diffusion-3.5-medium}"
PORT="${PORT:-8091}"

echo "Starting vLLM-Omni diffusion server..."
echo "Model: $MODEL"
echo "Port: $PORT"

vllm serve "$MODEL" --omni \
  --port "$PORT"

