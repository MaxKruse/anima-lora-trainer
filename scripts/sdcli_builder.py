"""Build sd-cli inference commands for LoRA evaluation."""

from pathlib import Path

# Anima-specific inference defaults (per Civitai developer docs + kohya-ss):
#   cfg-scale 3-5 (sweet spot 4), steps 25-35 (default 30), euler sampler, simple schedule
ANIMA_CFG_SCALE = "4.0"
ANIMA_STEPS = "30"
ANIMA_SAMPLER = "euler"
ANIMA_SCHEDULER = "simple"
ANIMA_NEGATIVE_PROMPT = "worst quality, low quality, blurry, bad anatomy, deformed hands"


def build_sdcli_command(
    diffusion_model: str,
    vae_model: str,
    llm_model: str,
    lora_file: str,
    prompt: str,
    seed: int,
    output_path: str,
    negative_prompt: str = ANIMA_NEGATIVE_PROMPT,
) -> str:
    """Assemble a shell command string for ``sd-cli`` inference with a LoRA.

    Returns a single string suitable for passing to ``subprocess.run``
    (on POSIX) or ``subprocess.run(..., shell=True)``.
    """
    lora_path = Path(lora_file)
    lora_dir = lora_path.parent.as_posix()
    lora_name = lora_path.name

    lora_prompt = f"<lora:{lora_name}:1> {prompt}"

    parts = [
        "sd-cli",
        "--model", diffusion_model,
        "--vae", vae_model,
        "--llm", llm_model,
        "--lora-model-dir", lora_dir,
        "--prompt", lora_prompt,
        "--negative-prompt", negative_prompt,
        "--cfg-scale", ANIMA_CFG_SCALE,
        "--sampling-method", ANIMA_SAMPLER,
        "--scheduler", ANIMA_SCHEDULER,
        "--steps", ANIMA_STEPS,
        "--diffusion-fa",
        "--offload-to-cpu",
        "-s", str(seed),
        "-o", output_path,
    ]

    return " ".join(f'{p}' if ' ' not in p else f'"{p}"' for p in parts)
