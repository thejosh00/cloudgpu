# CloudGPU Install

CLI tool to install Python apps on Lambda Labs cloud instances. Apps are installed on persistent storage and can be restored to a new instance with one command.

## Install

```bash
uv tool install -e .
```

This makes `cloudgpu` available globally. Requires Python 3.10+ locally. No dependencies needed on the instance (remote scripts use stdlib only).

## Quick Start

```bash
# 1. Set up a new instance (tests SSH, detects persistent storage, syncs tool)
cloudgpu setup ubuntu@<instance-ip>

# 2. Install ComfyUI
cloudgpu install --app comfyui

# 3. Launch it (SSH in and run the launch script)
cloudgpu ssh -- comfyui
# ComfyUI starts on 127.0.0.1:8188 (loopback only, on the instance)

# 3b. Reach the UI from your machine over an SSH tunnel
cloudgpu forward --port 8188
# then open http://localhost:8188
# (or skip step 3 and run `cloudgpu forward --run comfyui` to start ComfyUI AND tunnel it)

# 4. Instance terminated? Start a new one with the same storage, then:
cloudgpu recover <new-ip>

# 5. Check what's installed
cloudgpu status
```

After `setup`, the host is saved locally so subsequent commands can omit it.

## Commands

| Command | Description |
|---|---|
| `cloudgpu setup <host>` | Test SSH, detect persistent dir, sync tool, save config |
| `cloudgpu install [host] [--app comfyui]` | Install an app (interactive selection if no `--app`) |
| `cloudgpu recover [host]` | Restore everything on a new instance |
| `cloudgpu status [host]` | Show installed apps and their health |
| `cloudgpu ssh [-H host] [-- command]` | SSH wrapper with launch scripts on PATH |
| `cloudgpu forward [-H host] [-p port] [--local-port n] [--run [cmd]]` | Tunnel a remote port (default 8188/ComfyUI) to localhost; `--run` also starts the app over the same connection |
| `cloudgpu lambda ...` | Manage Lambda Cloud resources via the API (see below) |

## Lambda Cloud API

Provision instances and filesystems directly via the [Lambda Cloud API](https://docs-api.lambda.ai/api/cloud). Set your API key first ([create one here](https://cloud.lambda.ai/api-keys)):

```bash
export LAMBDA_API_KEY=secret_...
```

| Command | Description |
|---|---|
| `cloudgpu lambda instances` | List running instances |
| `cloudgpu lambda instance-types [--available]` | List instance types, pricing, and capacity |
| `cloudgpu lambda launch --region <r> --type <t> --ssh-key <k> [--filesystem <fs>] [--name <n>]` | Launch an instance |
| `cloudgpu lambda restart <id>...` | Restart instance(s) |
| `cloudgpu lambda terminate <id>... [-y]` | Terminate instance(s) |
| `cloudgpu lambda filesystems` | List filesystems |
| `cloudgpu lambda create-filesystem <name> --region <r>` | Create a filesystem |
| `cloudgpu lambda delete-filesystem <id> [-y]` | Delete a filesystem |

```bash
# Find an available GPU, launch it with a filesystem, then set up cloudgpu on it
cloudgpu lambda instance-types --available
cloudgpu lambda launch --region us-tx-1 --type gpu_1x_a10 --ssh-key my-key --filesystem my-fs
cloudgpu lambda instances          # grab the IP once it's active
cloudgpu setup ubuntu@<instance-ip>
```

## How It Works

1. Local CLI (your Mac) connects via SSH using your existing keys/config
2. `rsync` copies lightweight remote scripts to persistent storage
3. SSH runs `python3` on those scripts to install/recover/check status
4. Venvs use `--system-site-packages` to inherit Lambda's pre-installed PyTorch + CUDA (avoids re-downloading ~2GB)
5. `state.json` on persistent storage tracks what's installed
6. Recovery verifies files, checks PyTorch/CUDA compat, regenerates launch scripts

## Persistent Storage Layout

```
/lambda/nfs/<filesystem>/
├── .cloudgpu/
│   ├── remote/          # Tool scripts (synced from local)
│   ├── state.json       # Tracks installed apps
│   └── bin/             # Launch scripts
├── apps/
│   └── comfyui/         # App source (git clone)
└── venvs/
    └── comfyui/         # Python venv
```

## Adding New Apps

1. Create `src/cloudgpu/remote/apps/newapp.py` implementing `AppInstaller`
2. Register in `src/cloudgpu/remote/__main__.py` `APP_INSTALLERS` dict
3. Add to `AVAILABLE_APPS` in `src/cloudgpu/cli.py`

## Development

```bash
uv venv .venv --python 3.12
uv pip install -e ".[dev]"
.venv/bin/pytest tests/ -v
```

## Design Decisions

- **SSH via subprocess** (not paramiko) - uses your existing SSH config/keys, zero extra deps
- **rsync for sync** - efficient delta transfers, available on both Mac and Ubuntu
- **Remote scripts are stdlib-only** - no pip install needed on the instance
- **`--system-site-packages` venvs** - guaranteed CUDA compatibility with Lambda's PyTorch
- **State in JSON** - simple, human-readable, easy to debug
- **Lambda API via stdlib `urllib`** - no `requests`/`httpx` dependency; key read from `LAMBDA_API_KEY`
