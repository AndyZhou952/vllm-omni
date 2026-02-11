#!/bin/bash
# Deterministic LoRA sweep for /v1/images/generations.

set -euo pipefail

SERVER="${SERVER:-http://localhost:1997}"
PROMPT="${PROMPT:-MIA_char, standing in a new york city}"
SIZE="${SIZE:-1024x1024}"
SEED="${SEED:-42}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-15}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
LORA_NAME="${LORA_NAME:-SD3.5M-FlowGRPO-GenEval}"
LORA_PATH="${LORA_PATH:-/home/andy/model/SD3.5M-FlowGRPO-GenEval}"
OUT_DIR="${OUT_DIR:-lora_verify_output}"

mkdir -p "$OUT_DIR"

request_and_save() {
  local payload="$1"
  local output="$2"
  curl -s -X POST "$SERVER/v1/images/generations" \
    -H "Content-Type: application/json" \
    -d "$payload" | jq -r '.data[0].b64_json' | base64 -d > "$output"
}

echo "Generating deterministic baseline and LoRA sweep..."
echo "Server: $SERVER"
echo "Prompt: $PROMPT"
echo "LoRA path: $LORA_PATH"
echo "Output dir: $OUT_DIR"

request_and_save "{
  \"prompt\": \"$PROMPT\",
  \"size\": \"$SIZE\",
  \"seed\": $SEED,
  \"num_inference_steps\": $NUM_INFERENCE_STEPS,
  \"guidance_scale\": $GUIDANCE_SCALE
}" "$OUT_DIR/baseline.png"

request_and_save "{
  \"prompt\": \"$PROMPT\",
  \"size\": \"$SIZE\",
  \"seed\": $SEED,
  \"num_inference_steps\": $NUM_INFERENCE_STEPS,
  \"guidance_scale\": $GUIDANCE_SCALE,
  \"lora\": {
    \"name\": \"$LORA_NAME\",
    \"local_path\": \"$LORA_PATH\",
    \"scale\": 0.0
  }
}" "$OUT_DIR/lora_scale_0.png"

request_and_save "{
  \"prompt\": \"$PROMPT\",
  \"size\": \"$SIZE\",
  \"seed\": $SEED,
  \"num_inference_steps\": $NUM_INFERENCE_STEPS,
  \"guidance_scale\": $GUIDANCE_SCALE,
  \"lora\": {
    \"name\": \"$LORA_NAME\",
    \"local_path\": \"$LORA_PATH\",
    \"scale\": 0.5
  }
}" "$OUT_DIR/lora_scale_05.png"

request_and_save "{
  \"prompt\": \"$PROMPT\",
  \"size\": \"$SIZE\",
  \"seed\": $SEED,
  \"num_inference_steps\": $NUM_INFERENCE_STEPS,
  \"guidance_scale\": $GUIDANCE_SCALE,
  \"lora\": {
    \"name\": \"$LORA_NAME\",
    \"local_path\": \"$LORA_PATH\",
    \"scale\": 1.0
  }
}" "$OUT_DIR/lora_scale_1.png"

echo "Saved outputs:"
for f in "$OUT_DIR"/baseline.png "$OUT_DIR"/lora_scale_0.png "$OUT_DIR"/lora_scale_05.png "$OUT_DIR"/lora_scale_1.png; do
  if command -v sha256sum >/dev/null 2>&1; then
    echo "  $(sha256sum "$f")"
  else
    echo "  $(shasum -a 256 "$f")"
  fi
done

echo ""
echo "Next step:"
echo "  python verify_lora_online.py --server \"$SERVER\" --lora-path \"$LORA_PATH\" --out-dir \"$OUT_DIR\""
