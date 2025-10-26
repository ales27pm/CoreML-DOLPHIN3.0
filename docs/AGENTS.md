# Documentation Guidance

This scope governs all files under `docs/`.

- Treat documentation as code: changes require clear narrative structure, tables kept in sync
  with automation outputs, and explicit references to the manifest-driven workflow.
- Auto-managed sections must remain compatible with `tools/session_finalize.py`; avoid manual
  edits to generated timelines or snapshots.
- Provide contextual introductions for new documents and include cross-links back to the Codex
  ledger or roadmap when relevant.
- Use semantic headings (H1/H2/H3) and fenced code blocks to aid navigation.
- When adding templates, ensure they live under `docs/templates/` with descriptive names.
