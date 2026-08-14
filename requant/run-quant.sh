set -e
cd /work
echo "=== downloading base model ==="
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3.5-4B', local_dir='/work/Qwen3.5-4B')
print('DL_OK', p)
"
du -sh /work/Qwen3.5-4B
echo "=== quantizing (self_attn + linear_attn -> INT4) ==="
python3 /work/quant.py
echo "=== injecting MTP weights ==="
python3 /work/fork/inject_mtp_weights.py /work/Qwen3.5-4B /hostq/Qwen3.5-4B-AWQ-full
cp /hostq/Qwen3.5-4B-AWQ/chat_template.jinja /hostq/Qwen3.5-4B-AWQ-full/ 2>/dev/null || true
cp /hostq/Qwen3.5-4B-AWQ/preprocessor_config.json /hostq/Qwen3.5-4B-AWQ-full/ 2>/dev/null || true
cp /hostq/Qwen3.5-4B-AWQ/video_preprocessor_config.json /hostq/Qwen3.5-4B-AWQ-full/ 2>/dev/null || true
echo "=== new checkpoint quantization_config ==="
python3 -c "import json;print(json.dumps(json.load(open('/hostq/Qwen3.5-4B-AWQ-full/config.json')).get('quantization_config'),indent=1))"
echo "QUANT_ALL_DONE"
