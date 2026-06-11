"""Generate deterministic, compact folder names from permutation parameters."""

# Short codes for common parameter names
_PARAM_ABBR = {
    "batch_size": "bs",
    "epochs": "ep",
    "learning_rate": "lr",
    "mixed_precision": "mp",
    "network_alpha": "na",
    "network_dim": "nd",
    "optimizer": "opt",
    "resolution": "res",
    "scheduler": "sch",
    "timestep_sampling": "ts",
    "noise_offset": "no",
    "gradient_accumulation_steps": "gas",
    "max_train_steps": "mts",
    "seed": "seed",
    "clip_skip": "cs",
    "text_encoder_lr": "tlr",
    "unet_lr": "ulr",
    "lr_scheduler": "lrs",
    "lr_warmup_steps": "lrs_w",
    "rank_dropout": "rd",
    "dim": "d",
    "alpha": "a",
}

# Short codes for common parameter values
_VALUE_ABBR = {
    # Mixed precision
    "mixed-precision-bf16": "bf16",
    "mixed-precision-fp16": "fp16",
    "mixed-precision-none": "np",
    "bf16": "bf16",
    "fp16": "fp16",
    # Optimizers
    "adamw8bit": "a8b",
    "adamw": "aw",
    "adamw8bit-torch": "a8bt",
    "prodigy": "pr",
    "lion": "lion",
    "dadaptation": "da",
    # Schedulers
    "constant": "c",
    "cosine": "cos",
    "cosine-with-restarts": "cosr",
    "linear": "l",
    "polynomial": "poly",
    "constant-with-warmup": "cw",
    # Timestep sampling
    "sigmoid": "sig",
    "karras": "kar",
    "exponential": "exp",
    "uniform": "u",
}


def generate_folder_name(params: dict, prefix: str = "anima") -> str:
    """Generate a deterministic, compact folder name from permutation params.

    Uses short codes for parameter names and common values to keep paths
    well under Windows MAX_PATH (260) limits.

    Args:
        params: Dict of parameter names to values.
        prefix: Model prefix (default: "anima").

    Returns:
        Compact folder name like "anima_bs-4_ep-10_lr-1e-3_mp-bf16_na-1_nd-8_opt-a8b_res-1024_sch-c_ts-sig".
    """
    parts = [prefix]

    for key in sorted(params.keys()):
        value = params[key]
        formatted_value = _format_value(value)
        short_key = _short_param_name(key)
        parts.append(f"{short_key}-{formatted_value}")

    return "_".join(parts)


def _short_param_name(name: str) -> str:
    """Get a short code for a parameter name."""
    slug = name.lower().replace("_", "-")
    # Try exact match on slugified name
    if slug in _PARAM_ABBR:
        return _PARAM_ABBR[slug]
    # Try original name
    if name in _PARAM_ABBR:
        return _PARAM_ABBR[name]
    # Fallback: first 3 chars of each word joined
    words = name.lower().replace("-", " ").replace("_", " ").split()
    if len(words) == 1:
        return words[0][:4]
    return "".join(w[:2] for w in words)


def _format_value(value) -> str:
    """Format a parameter value for inclusion in a folder name.

    Uses short codes for common values.
    """
    if isinstance(value, bool):
        return "t" if value else "f"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        formatted = f"{value:e}"
        if "." in formatted:
            mantissa, exp = formatted.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            if exp.startswith("-"):
                exp = "-" + exp[1:].lstrip("0") or "0"
            else:
                exp = exp.lstrip("0") or "0"
            formatted = f"{mantissa}e{exp}" if not exp.startswith("-") else f"{mantissa}e-{exp[1:]}"
        return formatted
    else:
        str_val = str(value).lower().replace(" ", "-")
        if str_val in _VALUE_ABBR:
            return _VALUE_ABBR[str_val]
        return str_val


def _slugify(name: str) -> str:
    """Convert a parameter name to a URL-safe slug.

    "network_dim" -> "network-dim"
    """
    return name.lower().replace("_", "-")
