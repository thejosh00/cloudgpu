#!/usr/bin/env python3
"""Provisioning for this cloudgpu profile — downloads SDXL models.

Runs on every `cloudgpu up` (idempotent). Uses comfylib (the sibling comfylib.py).
Edit freely. Add Civitai models once CIVITAI_TOKEN is in ~/.config/cloudgpu/secrets.env.
"""
import comfylib as cl

# --- SDXL base + fp16 VAE (HuggingFace, public) ---
cl.huggingface(
    "stabilityai/stable-diffusion-xl-base-1.0",
    "sd_xl_base_1.0.safetensors",
    cl.checkpoints() / "sd_xl_base_1.0.safetensors",
)
cl.huggingface(
    "madebyollin/sdxl-vae-fp16-fix",
    "sdxl_vae.safetensors",
    cl.vae() / "sdxl_vae.safetensors",
)

# --- Civitai models (needs CIVITAI_TOKEN in secrets.env) ---
# Version id = the number in the model's download URL (/api/download/models/<id>).
# cl.civitai(128713, cl.checkpoints() / "dreamshaper_xl.safetensors")

# --- Custom nodes (optional) ---
# cl.custom_node("https://github.com/ltdrdata/ComfyUI-Impact-Pack")

print("Provisioning complete.")
