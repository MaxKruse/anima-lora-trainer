"""Build sd-cli inference commands for LoRA evaluation."""

from pathlib import Path


def build_sdcli_command(
    diffusion_model: str,
    vae_model: str,
    llm_model: str,
    lora_file: str,
    prompt: str,
    seed: int,
    output_path: str,
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
        "--cfg-scale", "6.0",
        "--sampling-method", "euler",
        "--steps", "20",
        "--diffusion-fa",
        "--offload-to-cpu",
        "-s", str(seed),
        "-o", output_path,
    ]

    return " ".join(f'{p}' if ' ' not in p else f'"{p}"' for p in parts)
