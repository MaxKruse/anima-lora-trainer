"""Generate deterministic folder names from permutation parameters."""


def generate_folder_name(params: dict, prefix: str = "anima") -> str:
    """Generate a deterministic folder name from permutation params.

    Args:
        params: Dict of parameter names to values.
        prefix: Model prefix (default: "anima").

    Returns:
        Folder name string like "anima_network-dim-32_network-alpha-16_learning-rate-1e-4".
    """
    parts = [prefix]

    # Sort params alphabetically for deterministic naming
    for key in sorted(params.keys()):
        value = params[key]
        formatted_value = _format_value(value)
        parts.append(f"{_slugify(key)}-{formatted_value}")

    return "_".join(parts)


def _format_value(value) -> str:
    """Format a parameter value for inclusion in a folder name.

    - Integers: plain string
    - Floats: scientific notation
    - Strings: lowercase, no spaces
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        # Use compact scientific notation (e.g., 1e-4 not 1.000000e-04)
        formatted = f"{value:e}"
        # Clean up: remove trailing zeros after decimal in mantissa
        if "." in formatted:
            mantissa, exp = formatted.split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            # Normalize exponent: remove leading zeros
            if exp.startswith("-"):
                exp = "-" + exp[1:].lstrip("0") or "0"
            else:
                exp = exp.lstrip("0") or "0"
            formatted = f"{mantissa}e{exp}" if not exp.startswith("-") else f"{mantissa}e-{exp[1:]}"
        return formatted
    else:
        return str(value).lower().replace(" ", "-")


def _slugify(name: str) -> str:
    """Convert a parameter name to a URL-safe slug.

    "network_dim" → "network-dim"
    """
    return name.lower().replace("_", "-")
