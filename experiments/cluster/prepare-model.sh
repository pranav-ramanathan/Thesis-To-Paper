#!/usr/bin/env bash
# Run once on a network-enabled setup host, then all compute jobs are offline.
set -euo pipefail
HP_SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HP_SCRIPT_DIR/environment.sh"
[[ $# -eq 1 ]] || { echo 'Usage: prepare-model.sh MODEL_DIR' >&2; exit 2; }
"$HP_VENV_PATH/bin/python" - "$1" <<'PY'
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
revision='45bb4654a4d5aaff24dd11d4781fa46d39bf8c13'
out=Path(sys.argv[1]).resolve()
snapshot_download('answerdotai/ModernBERT-large',revision=revision,local_dir=out,
                  allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json'])
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
files={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='cluster_pin.json'}
(out/'cluster_pin.json').write_text(json.dumps(dict(model='answerdotai/ModernBERT-large',revision=revision,files=files),indent=2)+'\n')
print(out)
PY
