# Source files in the GitHub edition

This directory retains the pinned configuration, tokenizer metadata and source
model card. `tokenizer.json` and the original weight shards are omitted from
GitHub. Their identities remain in `../source-verification.json`.

For inference, download `ane-lut6-c2048/models/**` from
`ales27pm/Dolphin3.0-CoreML` using the [version guide](../../README.md). That
directory includes the complete matching tokenizer. From the version root, use
`toolkit/tokenize_prompt.py --model-dir models --prompt 'Hello'` in the tokenizer
environment described in the toolkit guide.

To reproduce conversion, install `../requirements.txt` in the dedicated
environment and run this from the `toolkit/` directory:

```sh
.venv/bin/python download_source.py --source source
```

The downloader checks the pinned revision's identities, restores the missing
tokenizer, and downloads the original checkpoint shards. Downloaded files are
ignored by this version's Git rules.
