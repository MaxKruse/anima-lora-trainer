"""Constants and defaults for LoRA training."""

from pathlib import Path

# ── Project root ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "network_dim": 20,
    "network_alpha": 1,
    "learning_rate": 0.0002,
    "epochs": 2,
    "batch_size": 4,
    "max_steps": 800,
    "optimizer": "AdamW8Bit",
    "scheduler": "cosine",
    "resolution": 1024,
    "mixed_precision": "bf16",
    "timestep_sampling": "sigmoid",
    "gradient_checkpointing": True,
    "cache_latents": True,
    "cache_text_encoder": False,
    "caption_tag_dropout_rate": 0.05,
    "keep_tokens": 1,
}

# ── Validation limits ────────────────────────────────────────────────────
MIN_IMAGES_PER_FOLDER = 10
MAX_IMAGES_PER_FOLDER = 25
MIN_IMAGES_PER_OUTFIT = 15
MAX_IMAGES_BASE = 25

VALIDATION_MARKER = ".validation.json"

# ── Model paths ──────────────────────────────────────────────────────────
MODEL_PATHS = {
    "diffusion_model": "models/diffusion_model/anima-base-v1.0.safetensors",
    "vae": "models/vae/qwen_image_vae.safetensors",
    "text_encoder": "models/text_encoder/qwen_3_06b_base.safetensors",
}

# ── Image extensions ─────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
