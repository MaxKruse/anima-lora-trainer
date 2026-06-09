"""Tests for sd-cli command assembly."""

from scripts.sdcli_builder import build_sdcli_command


class TestBuildSdcliCommand:
    """Test construction of the sd-cli inference command."""

    def test_produces_command_with_correct_model_paths(self):
        """Command includes diffusion model, VAE, and LLM paths."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "models/anima/diffusion.safetensors" in cmd
        assert "models/anima/vae.safetensors" in cmd
        assert "models/anima/text_encoder/model.safetensors" in cmd

    def test_includes_lora_model_dir(self):
        """Command includes --lora-model-dir pointing to the permutation folder."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "--lora-model-dir" in cmd
        assert "outputs/run1/perm-a" in cmd

    def test_includes_prompt_with_lora_syntax(self):
        """Prompt is wrapped with <lora:{filename}:1> syntax."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "<lora:my_lora-000001.safetensors:1>" in cmd

    def test_sets_cfg_scale(self):
        """Command includes --cfg-scale 6.0."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "--cfg-scale" in cmd
        assert "6.0" in cmd

    def test_sets_sampling_method_and_steps(self):
        """Command includes --sampling-method euler and --steps 20."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "--sampling-method" in cmd
        assert "euler" in cmd
        assert "--steps" in cmd
        assert "20" in cmd

    def test_sets_diffusion_fa_and_offload_to_cpu(self):
        """Command includes --diffusion-fa and --offload-to-cpu flags."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "--diffusion-fa" in cmd
        assert "--offload-to-cpu" in cmd

    def test_sets_fixed_seed(self):
        """Command includes -s {seed} for deterministic output."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=12345,
            output_path="outputs/run1/eval_output.png",
        )

        assert "-s" in cmd
        assert "12345" in cmd

    def test_sets_output_path(self):
        """Command includes -o {output_path} for output image."""
        cmd = build_sdcli_command(
            diffusion_model="models/anima/diffusion.safetensors",
            vae_model="models/anima/vae.safetensors",
            llm_model="models/anima/text_encoder/model.safetensors",
            lora_file="outputs/run1/perm-a/my_lora-000001.safetensors",
            prompt="cat dog bird",
            seed=42,
            output_path="outputs/run1/eval_output.png",
        )

        assert "-o" in cmd
        assert "outputs/run1/eval_output.png" in cmd
