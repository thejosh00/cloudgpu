# CloudGPU

Declaratively manage GPU machines on Lambda Cloud. Describe the machine you want in a profile — GPU preference order, a persistent filesystem, and the apps to run — and `cloudgpu up` finds capacity, launches it, sets it up, and installs (or recovers) your apps. Because apps and data live on the persistent filesystem, a terminated instance is routine: re-run `cloudgpu up` and everything comes back on a fresh GPU.

## Install

```bash
uv tool install -e .
```

This makes `cloudgpu` available globally. Requires Python 3.10+ locally. No dependencies needed on the instance (remote scripts use stdlib only).

## Quick Start (profiles)

A **profile** describes the machine you want — which GPU (in preference order), which
filesystem holds its data, and which apps to keep installed. `cloudgpu up` then makes it so:
it finds capacity and launches the right GPU, sets the instance up, and installs/recovers
your apps. Run it again any time the instance has been terminated to bring everything back.

```bash
# 0. Export your Lambda Cloud API key (create one at https://cloud.lambda.ai/api-keys)
export LAMBDA_API_KEY=secret_...

# 1. Describe the machine you want (saved under ~/.config/cloudgpu/profiles/, per-user)
cloudgpu profile create comfyui --ssh-key my-key --gpu gh200,a100 --apps comfyui

# 2. Bring it up: poll for a GH200 (fallback A100), create a filesystem if needed,
#    launch, set up, install/recover comfyui.
#    Idempotent — re-run after a termination to recover onto a fresh instance.
cloudgpu up

# 3. Reach the UI: start ComfyUI and tunnel it in one command
cloudgpu forward --run comfyui
# then open http://localhost:8188

# 4. Done for now? Terminate the instance (data is kept on the filesystem):
cloudgpu down
#    ...or tear down everything including the filesystem and its data:
cloudgpu down --delete-filesystem

# Manage profiles
cloudgpu profile list          # the active one is marked *
cloudgpu profile use <name>    # switch the active profile (commands default to it)
cloudgpu profile edit          # edit the active profile's TOML in $EDITOR
```

`cloudgpu` manages the persistent filesystem for you: on the first `up` it creates one
(named after the profile, in whichever region first has GPU capacity) and reuses it on
every later `up`. Set `filesystem` in the profile to name it yourself or to reuse an
existing one.

### Provisioning (extra setup, e.g. model downloads)

For setup beyond installing the app — downloading models, installing custom nodes, etc. —
create a directory `~/.config/cloudgpu/profiles/<name>.provision/` with an entry point
(`provision.py` preferred, or `provision.sh`) plus any files it needs. For ComfyUI, copy
`scripts/comfylib.py` from this repo in alongside it:

```
~/.config/cloudgpu/profiles/
└── comfyui.provision/
    ├── provision.py         # entry point (provision.py preferred, else provision.sh)
    └── comfylib.py          # copied from scripts/comfylib.py — reusable helpers
```

On every `up`, cloudgpu rsyncs the whole directory to the instance and runs the entry
point from inside it (so it can `import comfylib` and reference sibling files). Make it
**idempotent** (download a model only if missing — `comfylib` does this for you) and write
data **under the filesystem** so it persists across terminations.

The entry point runs with its directory as CWD and these env vars (plus `cloudgpu/bin` on
`PATH`): `CLOUDGPU_PROVISION_DIR` (where the files landed), `CLOUDGPU_PERSISTENT_DIR`,
`CLOUDGPU_APPS_DIR`, `CLOUDGPU_VENVS_DIR`, `CLOUDGPU_BIN_DIR`. Raise the time budget with
`provision_timeout = 7200` (seconds) in the profile.

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

Profiles are per-user (not checked in). You can keep several; more than one can run at once,
as long as each uses its own filesystem (a filesystem backs only one instance at a time).
A second profile (different filesystem) runs concurrently:

```bash
cloudgpu profile create sd --ssh-key my-key --gpu a10 --apps comfyui   # filesystem "sd"
cloudgpu up --profile sd
```

An example `~/.config/cloudgpu/profiles/washington.toml`:

```toml
gpu = ["gh200", "a100"]       # GPU preference order (alias or full type name)
apps = ["comfyui"]            # apps to keep installed
ssh_key = "my-key"            # required: Lambda SSH key name (matching key in ~/.ssh)
# filesystem = "comfyui"      # optional; defaults to the profile name, auto-created on first up
poll_seconds = 20             # capacity poll interval
max_hours = 12                # give up after this long
```

## Manual workflow (low-level)

The individual steps `up` orchestrates are also available on their own:

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

Commands that target an instance (`install`, `recover`, `status`, `ssh`, `forward`) default
to the active profile's running instance; pass `-P/--profile <name>` to target another, or an
explicit host to bypass profiles entirely.

| Command | Description |
|---|---|
| `cloudgpu up [-P profile]` | Converge a profile's machine to its desired state: find the GPU, create the filesystem if needed, launch, set up, install/recover apps. Idempotent — re-run to recover after a termination |
| `cloudgpu down [-P profile] [--delete-filesystem] [-y]` | Terminate the profile's instance (data kept); `--delete-filesystem` also deletes the filesystem and its data |
| `cloudgpu profile create <name> --ssh-key <key> [--filesystem <fs>] [--gpu gh200,a100] [--apps comfyui]` | Scaffold a profile and select it (filesystem defaults to the profile name) |
| `cloudgpu profile list / show / use / edit / delete` | Manage profiles and the active selection |
| `cloudgpu setup <host>` | Test SSH, detect persistent dir, sync tool, save config |
| `cloudgpu install [host] [-P profile] [--app comfyui]` | Install an app (interactive selection if no `--app`) |
| `cloudgpu recover [host] [-P profile]` | Restore everything on a new instance |
| `cloudgpu status [host] [-P profile]` | Show installed apps and their health |
| `cloudgpu ssh [-H host] [-P profile] [-- command]` | SSH wrapper with launch scripts on PATH |
| `cloudgpu forward [-H host] [-P profile] [-p port] [--local-port n] [--run [cmd]]` | Tunnel a remote port (default 8188/ComfyUI) to localhost; `--run` also starts the app over the same connection |
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
