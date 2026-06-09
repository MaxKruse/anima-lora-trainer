"""Training command builder for kohya-ss scripts."""

from typing import Any


# Default model paths by model type
MODEL_PATHS = {
    "anima": {
        "diffusion_model": "models/anima/diffusion_models/anima-base-v1.0.safetensors",
        "vae": "models/anima/vae/qwen_image_vae.safetensors",
        "text_encoder": "models/anima/text_encoders/qwen_3_06b_base.safetensors",
        "train_script": "sd-scripts/anima_train_network.py",
        "network_module": "networks.lora_anima",
    },
}


def build_training_command(params: dict[str, Any]) -> list[str]:
    """Build the full accelerate launch command from training parameters.

    Args:
        params: Dictionary of training parameters.

    Returns:
        List of command arguments for shell execution.
    """
    model_type = params.get("model_type", "anima")
    paths = MODEL_PATHS.get(model_type)
    if not paths:
        raise ValueError(f"Unknown model type: {model_type}")

    cmd = [
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process", "1",
        paths["train_script"],
        # Model paths
        "--pretrained_model_name_or_path", paths["diffusion_model"],
        "--qwen3", paths["text_encoder"],
        "--vae", paths["vae"],
        # Dataset
        "--dataset_config", params["dataset_config"],
        # Output
        "--output_dir", params["output_dir"],
        f"--output_name={params['lora_name']}",
        "--save_model_as=safetensors",
        # Network
        "--network_module=" + paths["network_module"],
        f"--network_dim={params['network_dim']}",
        f"--network_alpha={params['network_alpha']}",
        # Training
        f"--learning_rate={params['learning_rate']}",
        f"--train_batch_size={params['batch_size']}",
        f"--max_train_epochs={params['epochs']}",
        f"--optimizer_type={params['optimizer']}",
        f"--lr_scheduler={params['scheduler']}",
        # Anima-specific
        f"--timestep_sampling={params['timestep_sampling']}",
        "--discrete_flow_shift=1.0",
        # Precision
        f"--mixed_precision={params['mixed_precision']}",
        # Optimizations
    ]

    if params.get("gradient_checkpointing"):
        cmd.append("--gradient_checkpointing")

    if params.get("cache_latents"):
        cmd.append("--cache_latents")

    if params.get("cache_text_encoder"):
        cmd.append("--cache_text_encoder_outputs")

    # VAE settings
    cmd.extend([
        "--vae_chunk_size=64",
        "--vae_disable_cache",
        "--save_every_n_epochs=1",
    ])

    return cmd
