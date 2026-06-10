"""Tests for training command builder."""

import pytest
from scripts.command_builder import build_training_command


class TestBuildTrainingCommand:
    """Test training command assembly from parameters."""

    def test_produces_command_with_all_required_flags(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/path/to/images",
            "lora_name": "my-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/outputs/test",
            "dataset_config": "/tmp/dataset.toml",
        })

        assert isinstance(cmd, list)
        # Should start with accelerate launch
        assert cmd[0] == "accelerate"
        assert cmd[1] == "launch"

    def test_includes_correct_model_paths(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        cmd_str = " ".join(cmd)
        assert "models/diffusion_model/anima-base-v1.0.safetensors" in cmd_str
        assert "models/vae/qwen_image_vae.safetensors" in cmd_str
        assert "models/text_encoder/qwen_3_06b_base.safetensors" in cmd_str

    def test_sets_network_module_lora_anima(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--network_module=networks.lora_anima" in cmd

    def test_includes_mixed_precision_bf16(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--mixed_precision=bf16" in cmd

    def test_includes_gradient_checkpointing(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--gradient_checkpointing" in cmd

    def test_includes_cache_latents(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--cache_latents" in cmd

    def test_includes_cache_text_encoder_outputs(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--cache_text_encoder_outputs" in cmd

    def test_includes_timestep_sampling_sigmoid(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--timestep_sampling=sigmoid" in cmd

    def test_includes_discrete_flow_shift(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--discrete_flow_shift=1.0" in cmd

    def test_includes_vae_chunk_size_and_disable_cache(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--vae_chunk_size=64" in cmd
        assert "--vae_disable_cache" in cmd

    def test_includes_save_n_epoch_ratio_when_using_epochs(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        # Epoch mode: save every 10% of epochs
        assert "--save_n_epoch_ratio=10" in cmd
        assert "--max_train_epochs=10" in cmd
        # Should NOT include max_train_steps or save_every_n_steps
        cmd_str = " ".join(cmd)
        assert "--max_train_steps" not in cmd_str
        assert "--save_every_n_steps" not in cmd_str

    def test_includes_save_every_n_steps_when_using_max_steps(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "max_steps": 500,
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        # Step mode: save every 10% of total steps (500/10 = 50)
        assert "--max_train_steps=500" in cmd
        assert "--save_every_n_steps=50" in cmd
        # Should NOT include epochs (mutually exclusive)
        cmd_str = " ".join(cmd)
        assert "--max_train_epochs" not in cmd_str
        assert "--save_n_epoch_ratio" not in cmd_str

    def test_save_interval_min_for_small_step_counts(self):
        cmd = build_training_command({
            "network_dim": 8,
            "network_alpha": 1,
            "learning_rate": 1e-4,
            "batch_size": 1,
            "epochs": 10,
            "max_steps": 5,  # Very small — interval should clamp to 1
            "optimizer": "AdamW8Bit",
            "scheduler": "cosine",
            "training_images": "/images",
            "lora_name": "test",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--save_every_n_steps=1" in cmd

    def test_includes_network_dim_and_alpha(self):
        cmd = build_training_command({
            "network_dim": 16,
            "network_alpha": 4,
            "learning_rate": 5e-4,
            "batch_size": 2,
            "epochs": 20,
            "optimizer": "Prodigy",
            "scheduler": "constant",
            "training_images": "/images",
            "lora_name": "custom-lora",
            "mixed_precision": "bf16",
            "timestep_sampling": "sigmoid",
            "gradient_checkpointing": True,
            "cache_latents": True,
            "cache_text_encoder": True,
            "output_dir": "/out",
            "dataset_config": "/tmp/d.toml",
            "model_type": "anima",
        })

        assert "--network_dim=16" in cmd
        assert "--network_alpha=4" in cmd
        assert "--learning_rate=0.0005" in cmd
        assert "--train_batch_size=2" in cmd
        assert "--max_train_epochs=20" in cmd
        assert "--optimizer_type=Prodigy" in cmd
        assert "--lr_scheduler=constant" in cmd
        assert "--output_name=custom-lora" in cmd
