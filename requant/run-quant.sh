set -e
Q=/qwork
mkdir -p "$Q"
echo "=== [1/5] pip install (AutoAWQ + flash-linear-attention) ==="
pip install --no-cache-dir "git+https://github.com/casper-hansen/AutoAWQ" flash-linear-attention 2>&1 | tail -1

echo "=== [2/5] apply autoawq-qwen35 fork ==="
[ -d "$Q/fork" ] || git clone --depth 1 https://github.com/quivent/autoawq-qwen35 "$Q/fork"
A=/usr/local/lib/python3.12/dist-packages/awq
cp "$Q/fork/qwen3_5.py" "$A/models/qwen3_5.py"
# VARIANT D: FULL quant config — self_attn + GDN qkv/z/out + MLP INT4,
# in_proj_a/b fp16 (fork default). The class attribute modules_to_not_convert
# OVERRIDES the user quant_config (awq/models/base.py:222 assigns it
# unconditionally), so the fork's hardcoded list must be rewritten here to
# exactly the fork default (a no-op rewrite — machinery + guard kept so a fork
# drift is caught). The sed re-runs each attempt because line 10 re-copies the
# pristine fork file over the installed one.
sed -i 's/modules_to_not_convert = \["visual", "mtp", "in_proj_b", "in_proj_a"\]/modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]/' "$A/models/qwen3_5.py"
grep -qF 'modules_to_not_convert = ["visual", "mtp", "in_proj_b", "in_proj_a"]' "$A/models/qwen3_5.py" || { echo "MTc_OVERRIDE_FAILED"; exit 1; }
echo "mtc override OK: $(grep -nF 'modules_to_not_convert = ' "$A/models/qwen3_5.py" | head -1)"
cd "$A/.."
for p in __init__.py.patch auto.py.patch quantizer.py.patch; do
  patch -p1 -N < "$Q/fork/$p" 2>&1 | tail -1 || true
done
# repair the fuzzy-patch corruption (two imports merged onto one line)
grep -q "Generationfrom" "$A/models/__init__.py" && \
  sed -i "s/Generationfrom .qwen3_5 import/Generation\nfrom .qwen3_5 import/" "$A/models/__init__.py"
# base.py.patch in the fork is a hand-written pseudo-patch: apply its intent manually
grep -q "AutoModelForImageTextToText" "$A/models/base.py" || \
  sed -i '/"qwen2_5_omni": "AutoModelForTextToWaveform",/a\    "qwen3_5": "AutoModelForImageTextToText",' "$A/models/base.py"
# use_cache exclusion (fork base.py.patch, second hunk)
grep -q 'AutoModelForImageTextToText")):' "$A/models/base.py" || \
  sed -i 's/== "AutoModelForTextToWaveform")):/== "AutoModelForTextToWaveform") or (target_cls_name == "AutoModelForImageTextToText")):/' "$A/models/base.py"
python3 -c "import awq" 2>/dev/null && echo "awq imports OK" || { echo "AWQ_INSTALL_FAILED"; exit 1; }

echo "=== [3/5] download base model (disk-backed, resumable) ==="
export HF_HOME="$Q/hf"
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3.5-4B', local_dir='$Q/Qwen3.5-4B')
print('DL_OK', p)
"

echo "=== [4/5] VARIANT D: FULL config (self_attn + GDN qkv/z/out + MLP INT4; in_proj_a/b fp16-compensated) ==="
python3 /qwork/quant.py

echo "=== [5/5] MTP inject + config fixups ==="
python3 "$Q/fork/inject_mtp_weights.py" "$Q/Qwen3.5-4B" /hostq/Qwen3.5-4B-AWQ-vd
for f in chat_template.jinja preprocessor_config.json video_preprocessor_config.json; do
  cp "/hostq/Qwen3.5-4B-AWQ/$f" /hostq/Qwen3.5-4B-AWQ-vd/ 2>/dev/null || true
done
python3 -c "import json;print(json.dumps(json.load(open('/hostq/Qwen3.5-4B-AWQ-vd/config.json')).get('quantization_config'),indent=1))"
echo "QUANT_ALL_DONE"
