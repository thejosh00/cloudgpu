# CloudGPU

Declaratively manage GPU machines on Lambda Cloud. Describe the machine you want in a profile — GPU preference order, a persistent filesystem, and optionally some apps — and `cloudgpu up` finds capacity, launches it, sets it up, and installs (or recovers) your apps. Because data lives on the persistent filesystem, a terminated instance is routine: re-run `cloudgpu up` and everything comes back on a fresh GPU.

## Install

```bash
uv tool install -e .
```

This makes `cloudgpu` available globally. Requires Python 3.10+ locally. No dependencies needed on the instance (remote scripts use stdlib only).

## Quick Start

A **profile is a folder** anywhere on disk. It holds the machine definition
(`cloudgpu.toml`), its provisioning, and its state. You `cd` into it and run `cloudgpu up`.

```bash
# 0. Export your Lambda Cloud API key (create one at https://cloud.lambda.ai/api-keys)
export LAMBDA_API_KEY=secret_...

# 1. Scaffold a profile folder.
mkdir my-comfy && cd my-comfy
cloudgpu init --ssh-key my-key --gpu gh200,a100 --apps comfyui

# 2. Bring it up: poll for a GH200 (fallback A100), create the filesystem if needed,
#    launch, set up, install/recover comfyui, run provisioning.
#    Idempotent — re-run after a termination to recover onto a fresh instance.
cloudgpu up

# 3. Reach the UI. ComfyUI runs as a service on the instance, so just open a tunnel:
cloudgpu forward     # forwards the profile's app ports (8188 for comfyui)
# then open http://localhost:8188

# 4. Done for now? Terminate the instance (data is kept on the filesystem):
cloudgpu down
#    ...or tear down everything including the filesystem and its data:
cloudgpu down --delete-filesystem
```

Commands act on the profile in the **current directory** (the one with a `cloudgpu.toml`);
pass `--profile <dir>` to point elsewhere. Runtime state lives in `cloudgpu.state.json` in
the folder (gitignored). Keep several profiles as separate folders — more than one can run
at once, as long as each uses its own filesystem (a filesystem backs one instance at a time).

`cloudgpu` manages the persistent filesystem for you: on the first `up` it creates one
(named after the folder, in whichever region first has GPU capacity) and reuses it on every
later `up`. Set `filesystem` in `cloudgpu.toml` to name it yourself or reuse an existing one.

Apps can run as **systemd services** on the instance. ComfyUI does: it binds `127.0.0.1:8188`,
auto-starts on boot, and restarts on crash — no need to launch it by hand. `cloudgpu forward`
opens a tunnel to each of the profile's app ports; `cloudgpu status` shows whether services
are active. Access a terminal with `cloudgpu ssh`.

### Provisioning (extra setup, e.g. model downloads)

Apps that need extra setup drop a `provision.py` (and helpers like `comfylib.py`) into the
profile folder at `cloudgpu init` time. On every `up`, the **whole folder** is rsynced to the
instance (excluding `cloudgpu.state.json`, `secrets.env`, and VCS/cache files) and
`provision.py` (or `provision.sh`) runs from inside it. Use it to download models, install
custom nodes, etc. — make it **idempotent** (download a model only if missing; `comfylib`
does this for you) and write data **under the filesystem**
so it persists across terminations. A profile with no provision script just skips this step.

```
my-comfy/
├── cloudgpu.toml         # the machine definition
├── provision.py          # entry point (or provision.sh); runs every `up`
├── comfylib.py           # reusable helpers, vendored by `init`
└── cloudgpu.state.json   # tool-written state (gitignored, never rsynced)
```

The entry point runs with the folder as CWD and these env vars (plus `cloudgpu/bin` on
`PATH`): `CLOUDGPU_PROVISION_DIR` (where the files landed), `CLOUDGPU_PERSISTENT_DIR`,
`CLOUDGPU_APPS_DIR`, `CLOUDGPU_VENVS_DIR`, `CLOUDGPU_BIN_DIR`. Raise the time budget with
`provision_timeout = 7200` (seconds) in `cloudgpu.toml`.

Example `provision.py` using `comfylib` — fetch SDXL and (optionally) a Civitai model:

```python
import comfylib as cl

# HuggingFace (public): SDXL base + fp16 VAE
cl.huggingface("stabilityai/stable-diffusion-xl-base-1.0",
               "sd_xl_base_1.0.safetensors", cl.checkpoints() / "sd_xl_base_1.0.safetensors")
cl.huggingface("madebyollin/sdxl-vae-fp16-fix",
               "sdxl_vae.safetensors", cl.vae() / "sdxl_vae.safetensors")

# Civitai (needs CIVITAI_TOKEN in secrets.env): version id from the model's download URL
# cl.civitai(128713, cl.checkpoints() / "dreamshaper_xl.safetensors")

# Custom nodes (optional):
# cl.custom_node("https://github.com/ltdrdata/ComfyUI-Impact-Pack")
```

`comfylib` downloads idempotently/resumably, resolves ComfyUI's model dirs
(`checkpoints()`, `vae()`, `loras()`, ...), and passes auth tokens to `curl` via stdin so
they never hit the process list. (A plain `provision.sh` still works for non-ComfyUI or
shell-only setups.)

#### Secrets (API tokens, e.g. Civitai)

For downloads that need a private token, **do not put it in the script**. Add it to
`~/.config/cloudgpu/secrets.env` (mode 600, `KEY=VALUE` per line):

```bash
CIVITAI_TOKEN=your_token_here
```

On `up`, cloudgpu transfers this file to the instance (as content, never on a command
line), sources it into the provision environment, and removes it afterward — so the
script references `$CIVITAI_TOKEN` and never hardcodes the value. Use a header so the
token stays out of URLs and logs:

```bash
curl -fL -C - -H "Authorization: Bearer $CIVITAI_TOKEN" \
  -o "$dest" "https://civitai.com/api/download/models/<versionId>"
```

`secrets.env` is never committed and never printed; the included `CLAUDE.md` instructs
Claude not to read it, so you can have Claude edit provision scripts without exposing the
value.

A second profile is just another folder (with its own filesystem); it runs concurrently:

```bash
mkdir ~/sd && cd ~/sd && cloudgpu init --ssh-key my-key --gpu a10 --apps comfyui
cloudgpu up
```

`cloudgpu.toml` looks like:

```toml
gpu = ["gh200", "a100"]       # GPU preference order (alias or full type name)
apps = ["comfyui"]            # apps to keep installed
ssh_key = "my-key"            # required: Lambda SSH key name (matching key in ~/.ssh)
# filesystem = "my-comfy"     # optional; defaults to the folder name, auto-created on first up
poll_seconds = 20             # capacity poll interval
max_hours = 12                # give up after this long
```

## Manual workflow (low-level)

The individual steps `up` orchestrates are also available on their own:

```bash
# 1. Set up a new instance (tests SSH, detects persistent storage, syncs tool)
cloudgpu setup ubuntu@<instance-ip>

# 2. Install ComfyUI (also sets up + starts the systemd service)
cloudgpu install --app comfyui

# 3. Reach the UI over an SSH tunnel (the service is already running on 127.0.0.1:8188)
cloudgpu forward --port 8188
# then open http://localhost:8188
# (to run it in the foreground for debugging instead: cloudgpu ssh -- comfyui)

# 4. Instance terminated? Start a new one with the same storage, then:
cloudgpu recover <new-ip>

# 5. Check what's installed
cloudgpu status
```

After `setup`, the host is saved locally so subsequent commands can omit it.

## Commands

Profile commands (`up`, `down`, `install`, `recover`, `status`, `ssh`, `forward`) act on the
`cloudgpu.toml` in the current directory; pass `--profile <dir>` to target another folder, or
(for the manual flow) an explicit host.

| Command | Description |
|---|---|
| `cloudgpu init [dir] --ssh-key <key> [--gpu gh200,a100] [--apps comfyui] [--filesystem <fs>]` | Scaffold a profile folder (cloudgpu.toml + .gitignore); each `--app` also vendors its files (comfyui → comfylib.py + provision.py). No `--apps` = bare machine |
| `cloudgpu up [-P dir]` | Converge the profile's machine: find the GPU, create the filesystem if needed, launch, set up, install/recover apps, provision. Idempotent — re-run to recover after a termination |
| `cloudgpu down [-P dir] [--delete-filesystem] [-y]` | Terminate the profile's instance (data kept); `--delete-filesystem` also deletes the filesystem and its data |
| `cloudgpu setup <host>` | Test SSH, detect persistent dir, sync tool, save config (manual flow) |
| `cloudgpu install [host] [-P dir] [--app comfyui]` | Install an app (interactive selection if no `--app`) |
| `cloudgpu recover [host] [-P dir]` | Restore everything on a new instance |
| `cloudgpu status [host] [-P dir]` | Show installed apps and their health |
| `cloudgpu ssh [-H host] [-P dir] [-- command]` | SSH wrapper with launch scripts on PATH |
| `cloudgpu forward [-P dir] [-p port] [--run [cmd]]` | Tunnel the profile's app port(s) to localhost (e.g. 8188 for ComfyUI); `-p` forwards a specific port; `--run` runs a command over the tunnel |
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

1. Local CLI connects via SSH using your existing keys/config
2. `rsync` copies lightweight remote scripts to persistent storage
3. SSH runs `python3` on those scripts to install/recover/check status
4. `state.json` on persistent storage tracks what's installed
5. Recovery verifies files, checks PyTorch/CUDA compat, regenerates launch scripts

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
- **State in JSON** - simple, human-readable, easy to debug
- **Lambda API via stdlib `urllib`** - no `requests`/`httpx` dependency; key read from `LAMBDA_API_KEY`
