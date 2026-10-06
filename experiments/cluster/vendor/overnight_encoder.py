"""Pinned, offline-only frozen sequence features for the overnight experiment."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import torch

ENCODER = "answerdotai/ModernBERT-large"
REVISION = "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13"
HERE = Path(__file__).resolve().parent
SNAPSHOT = HERE / ".cache/hub/models--answerdotai--ModernBERT-large/snapshots" / REVISION


def encode_sequences(sequences, out, max_n=64):
    """Return sequence -> FP32 CPU [max_n, hidden] features and provenance.

    Every HP residue must map to exactly one tokenizer word/token; special
    tokens are excluded. Existing caches are reused only on exact input,
    revision, and padding-length matches. Neither tokenizer nor model loading
    is allowed to access the network.
    """
    started = time.perf_counter()
    sequences = list(sequences)
    if not isinstance(max_n, int) or max_n < 2:
        raise ValueError("max_n must be an integer >= 2")
    if not sequences or len(set(sequences)) != len(sequences):
        raise ValueError("Provide a nonempty list of unique sequences")
    if any(not isinstance(s, str) or not 2 <= len(s) <= max_n or set(s) - {"H", "P"}
           for s in sequences):
        raise ValueError("Sequences must contain only H/P, with length 2..max_n")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "encoder_features.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True, map_location="cpu")
        if (cached["sequences"] != sequences or cached["revision"] != REVISION
                or cached.get("max_n") != max_n):
            raise ValueError("Existing encoder cache does not match requested inputs")
        features = cached["features"]
        if (features.ndim != 3 or features.shape[:2] != (len(sequences), max_n)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError("Invalid cached encoder feature shape, precision, or values")
        metadata = dict(cached["metadata"], cache_reused=True, model_load_count_this_call=0,
                        cache_load_s=time.perf_counter() - started)
        return {s: features[i] for i, s in enumerate(sequences)}, metadata

    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (SNAPSHOT / name).is_file():
            raise FileNotFoundError(f"Required pinned local encoder file missing: {SNAPSHOT / name}")
    # The workspace dependency target was installed earlier; this adds no
    # packages and does not import or mutate sibling project code.
    deps = str(HERE / ".deps")
    if deps not in sys.path:
        sys.path.insert(0, deps)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["USE_FLAX"] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    model = AutoModel.from_pretrained(str(SNAPSHOT), local_files_only=True,
                                     attn_implementation="sdpa", reference_compile=False)
    model = model.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)
    load_s = time.perf_counter() - load_started
    result = []
    encoding_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(sequences), 8):
            batch = sequences[start:start + 8]
            inputs = tokenizer([list(s) for s in batch], is_split_into_words=True,
                               return_tensors="pt", padding=True, truncation=False)
            outputs = model(**inputs).last_hidden_state
            for row, seq in enumerate(batch):
                word_ids = inputs.word_ids(row)
                if {w for w in word_ids if w is not None} != set(range(len(seq))):
                    raise ValueError("Tokenizer word IDs do not cover precisely the sequence")
                tensor = torch.zeros(max_n, outputs.shape[-1], dtype=torch.float32)
                for residue in range(len(seq)):
                    locations = [i for i, word in enumerate(word_ids) if word == residue]
                    if len(locations) != 1:
                        raise ValueError("Expected exactly one HP token per residue")
                    tensor[residue] = outputs[row, locations].mean(0)
                tensor = tensor / torch.sqrt(tensor.square().mean(-1, keepdim=True) + 1e-8)
                if not torch.isfinite(tensor).all():
                    raise ValueError("Nonfinite frozen encoder features")
                result.append(tensor)
            print(f"offline encoder: {min(start + 8, len(sequences))}/{len(sequences)} sequences", flush=True)
    encoding_s = time.perf_counter() - encoding_started
    features = torch.stack(result)
    metadata = dict(model=ENCODER, revision=REVISION, snapshot=str(SNAPSHOT), max_n=max_n,
                    parameters=sum(p.numel() for p in model.parameters()), frozen=True,
                    device="cpu", precision="float32", batch_size=8, torch_threads=4,
                    torch_version=str(torch.__version__), transformers_version=transformers.__version__,
                    model_load_count_this_call=1, cache_reused=False, local_files_only=True,
                    api_calls=0, downloads=0, model_and_tokenizer_load_s=load_s,
                    encoding_s=encoding_s, encoding_and_load_s=time.perf_counter() - started,
                    timing_scope="Includes dependency import, model/tokenizer load and encoding; excludes cache writing",
                    input="Full HP sequence as separate residue words; exactly one token feature per residue",
                    normalization="Per-residue divide by sqrt(mean(feature**2)+1e-8); zero padding",
                    head_input="Frozen sequence features plus explicit partial geometry and candidate features")
    temporary = cache_path.with_suffix(".pt.tmp")
    write_started = time.perf_counter()
    torch.save(dict(sequences=sequences, features=features, revision=REVISION,
                    max_n=max_n, metadata=metadata), temporary)
    temporary.replace(cache_path)
    # Returned provenance can account for cache writing separately without
    # rewriting an already durable feature cache to update its own timing.
    returned_metadata = dict(metadata, cache_write_s=time.perf_counter() - write_started,
                             helper_wall_s=time.perf_counter() - started)
    del model
    return {s: features[i] for i, s in enumerate(sequences)}, returned_metadata
