"""comfylib — helpers for ComfyUI cloudgpu provision scripts.

Copy this file into your profile's ``<name>.provision/`` directory next to ``provision.py``
and import it:

    import comfylib as cl

    # HuggingFace (public) — base SDXL + fp16 VAE
    cl.huggingface("stabilityai/stable-diffusion-xl-base-1.0",
                   "sd_xl_base_1.0.safetensors", cl.checkpoints() / "sd_xl_base_1.0.safetensors")

    # Civitai (needs CIVITAI_TOKEN from secrets.env)
    cl.civitai(128713, cl.checkpoints() / "dreamshaper_xl.safetensors")

Design notes:
- Stdlib only; downloads shell out to ``curl`` (resume + redirects), which is on the box.
- Idempotent: a file that already exists is skipped; partial downloads resume (.part).
- Secret-safe: auth tokens are passed to curl via a stdin config, never on argv (so they
  don't show up in the instance's process list).
- Reads the ``CLOUDGPU_*`` env vars that ``cloudgpu up`` sets, and writes everything under
  the persistent filesystem so it survives instance termination.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    print(f"comfylib: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def _log(msg: str) -> None:
    print(msg, flush=True)


# --- paths ----------------------------------------------------------------


def comfy_dir() -> Path:
    """The ComfyUI install dir (override with COMFYUI_DIR; else $CLOUDGPU_APPS_DIR/comfyui)."""
    override = os.environ.get("COMFYUI_DIR")
    if override:
        return Path(override)
    apps = os.environ.get("CLOUDGPU_APPS_DIR")
    if not apps:
        _fail("CLOUDGPU_APPS_DIR not set — run this via `cloudgpu up` (or set COMFYUI_DIR).")
    return Path(apps) / "comfyui"


def models_dir() -> Path:
    return comfy_dir() / "models"


def model_dir(kind: str) -> Path:
    """A ComfyUI models subdir (created), e.g. model_dir('checkpoints')."""
    d = models_dir() / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def checkpoints() -> Path: return model_dir("checkpoints")
def vae() -> Path: return model_dir("vae")
def loras() -> Path: return model_dir("loras")
def controlnet() -> Path: return model_dir("controlnet")
def upscale_models() -> Path: return model_dir("upscale_models")
def embeddings() -> Path: return model_dir("embeddings")
def clip() -> Path: return model_dir("clip")
def clip_vision() -> Path: return model_dir("clip_vision")
def unet() -> Path: return model_dir("unet")


# --- downloads ------------------------------------------------------------


def download(url: str, dest, *, headers: dict[str, str] | None = None) -> Path:
    """Download ``url`` to ``dest`` unless it already exists. Resumable and atomic.

    ``headers`` (e.g. an Authorization token) are passed to curl via a stdin config so
    they never appear on the command line / process list.
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        _log(f"  ✓ already present: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.parent / (dest.name + ".part")

    _log(f"  ↓ downloading {dest.name} ...")
    cmd = ["curl", "-fL", "-C", "-", "--retry", "3", "--retry-delay", "2", "-o", str(part)]
    stdin = None
    if headers:
        cmd += ["-K", "-"]  # read extra options (the secret header) from stdin
        stdin = "\n".join(f'header = "{k}: {v}"' for k, v in headers.items())
    cmd += [url]
    subprocess.run(cmd, check=True, input=stdin, text=True)

    part.replace(dest)
    _log(f"  ✓ done: {dest.name}")
    return dest


def huggingface(repo: str, filename: str, dest, *, revision: str = "main",
                token: str | None = None) -> Path:
    """Download a file from a HuggingFace repo. Set HF_TOKEN (secrets.env) for gated repos."""
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    token = token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    return download(url, dest, headers=headers)


def civitai(version_id: int | str, dest, *, token: str | None = None) -> Path:
    """Download a Civitai model *version* by id. Needs CIVITAI_TOKEN (from secrets.env).

    The version id is the number in the model page's download URL,
    https://civitai.com/api/download/models/<versionId>.
    """
    token = token or os.environ.get("CIVITAI_TOKEN")
    if not token:
        _fail("CIVITAI_TOKEN not set — add it to ~/.config/cloudgpu/secrets.env.")
    url = f"https://civitai.com/api/download/models/{version_id}"
    return download(url, dest, headers={"Authorization": f"Bearer {token}"})


# --- comfyui extras -------------------------------------------------------


def custom_node(git_url: str, *, name: str | None = None) -> Path:
    """Clone (or update) a ComfyUI custom node and install its requirements into the venv."""
    name = name or git_url.rstrip("/").split("/")[-1].removesuffix(".git")
    nodes = comfy_dir() / "custom_nodes"
    nodes.mkdir(parents=True, exist_ok=True)
    dest = nodes / name
    if dest.exists():
        _log(f"  ↻ updating custom node {name} ...")
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only"], check=False)
    else:
        _log(f"  ↓ cloning custom node {name} ...")
        subprocess.run(["git", "clone", "--depth", "1", git_url, str(dest)], check=True)

    req = dest / "requirements.txt"
    venvs = os.environ.get("CLOUDGPU_VENVS_DIR")
    if req.exists() and venvs:
        pip = Path(venvs) / "comfyui" / "bin" / "pip"
        if pip.exists():
            subprocess.run([str(pip), "install", "-r", str(req)], check=True)
    return dest
