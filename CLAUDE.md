# CLAUDE.md

## Workflow

- **Do not commit.** The user handles all git commits themselves. Make and verify
  code changes, but leave staging/committing to them unless explicitly asked.

## Secrets

- **Never read, open, cat, print, or echo `~/.config/cloudgpu/secrets.env`** (or any
  `*.secrets.env` file). It holds private API tokens (e.g. CIVITAI_TOKEN). You may
  reference the variable names in scripts (e.g. `$CIVITAI_TOKEN`) and edit provision
  scripts that use them, but never inspect or output the values. If you need to know
  whether it exists, check the file's presence — do not read its contents.
