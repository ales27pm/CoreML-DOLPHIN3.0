#!/usr/bin/env python3
"""Print exact pinned-tokenizer ChatML IDs for the Swift validation CLI."""
import argparse
import json
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model-dir", default="output")
p.add_argument("--system", default="You are Dolphin, a helpful AI assistant.")
p.add_argument("--prompt", required=True)
p.add_argument("--json", action="store_true")
a = p.parse_args()
t = AutoTokenizer.from_pretrained(a.model_dir, local_files_only=True)
ids = t.apply_chat_template([{"role": "system", "content": a.system}, {"role": "user", "content": a.prompt}],
                            tokenize=True, add_generation_prompt=True)
print(json.dumps(ids) if a.json else ",".join(map(str, ids)))
